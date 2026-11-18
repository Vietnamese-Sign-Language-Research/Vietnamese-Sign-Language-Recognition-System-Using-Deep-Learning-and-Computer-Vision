"""Protect the language-specific model routing and candidate release boundary."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
import torch

from letter_skeleton import LETTERS, SkeletonNet, SkeletonRecognizer, merge_predictions, merge_k_specialist
from recognition import ModelRegistry


def scores(label, confidence):
    p=np.full((1,len(LETTERS)),(1-confidence)/(len(LETTERS)-1),dtype='float32')
    p[0,LETTERS.index(label)]=confidence
    return p


class LetterReleaseTests(unittest.TestCase):
    def test_k_specialist_preserves_strong_corrected_h_and_r(self):
        for label in ['H','R']:
            np.testing.assert_equal(merge_k_specialist(scores(label,.95),[.99]),scores(label,.95))

    def test_k_specialist_requires_high_evidence_before_overriding(self):
        initial=scores('L',.7)
        np.testing.assert_equal(merge_k_specialist(initial,[.9]),initial)
        self.assertEqual(LETTERS[int(merge_k_specialist(initial,[.99]).argmax())],'K')

    def fallback_registry(self,label,boxes=None):
        registry=ModelRegistry()
        registry.letters=Mock()
        registry.letters.image_result.return_value={'status':'no_sign','label':None}
        registry.detectors['letters']=object()
        registry._detector_image_result=Mock(return_value={'status':'recognized','label':label,
            'details':{'score':.79,'boxes':boxes or []}})
        return registry

    def test_fallback_never_reintroduces_conflicting_labels(self):
        for label in ['H','K','R','D']:
            registry=self.fallback_registry(label)
            registry._detector_image_result.return_value['details']['score']=.99
            self.assertIsNone(registry.image_result(np.zeros((32,32,3),dtype='uint8'),'letters')['label'])

    def test_separate_agreeing_hands_can_preserve_small_photo_recognition(self):
        boxes=[{'label':'X','score':.79,'bounds':[.1,.1,.3,.8]},
               {'label':'X','score':.78,'bounds':[.7,.1,.9,.8]}]
        registry=self.fallback_registry('X',boxes)
        self.assertEqual(registry.image_result(np.zeros((32,32,3),dtype='uint8'),'letters')['label'],'X')

    def test_duplicate_photo_boxes_are_not_two_hand_consensus(self):
        box={'label':'X','score':.79,'bounds':[.1,.1,.3,.8]}
        registry=self.fallback_registry('X',[box,box])
        self.assertIsNone(registry.image_result(np.zeros((32,32,3),dtype='uint8'),'letters')['label'])

    def test_corrected_model_advertises_its_actual_vocabulary(self):
        registry=ModelRegistry()
        registry.letters=SimpleNamespace(labels=LETTERS,engine='vsl-corrected-ensemble-v1')
        model=next(m for m in registry.metadata() if m['id']=='letters')
        self.assertTrue(model['available'])
        self.assertEqual(model['count'],23)
        self.assertIn('Đ',model['labels'])
        self.assertEqual(model['engine'],'vsl-corrected-ensemble-v1')

    def test_geometry_head_cannot_restore_incompatible_k_or_r(self):
        for label in ['H','K','R']:
            cnn=scores('L',.6)
            result=merge_predictions(cnn,scores(label,.99))
            np.testing.assert_equal(result,cnn)

    def test_corrected_hkr_prediction_takes_precedence(self):
        for label in ['H','K','R']:
            result=merge_predictions(scores(label,.95),scores('U',.99))
            self.assertEqual(LETTERS[int(result.argmax())],label)

    def test_geometry_head_can_resolve_a_non_conflicting_letter(self):
        result=merge_predictions(scores('V',.6),scores('Y',.97))
        self.assertEqual(LETTERS[int(result.argmax())],'Y')

    def test_uncertain_corrected_kr_is_not_overridden_by_other_legacy_letters(self):
        for label in ['H','K','R']:
            cnn=scores(label,.4)
            np.testing.assert_equal(merge_predictions(cnn,scores('P',.99)),cnn)

    def test_unapproved_checkpoint_cannot_be_loaded_by_the_site(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'candidate.pt'
            torch.save(dict(format='vsl-skeleton-v1',labels=LETTERS,approved_for_production=False),path)
            with self.assertRaisesRegex(ValueError,'release checks'):
                SkeletonRecognizer(path)

    def test_demo_approval_requires_explicit_loader_opt_in(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'demo.pt'
            torch.save(dict(format='vsl-skeleton-v1',labels=LETTERS,state_dict=SkeletonNet().state_dict(),
                approved_for_production=False,approved_for_demo=True,release_channel='demo',experimental_labels=['H','K','R']),path)
            with self.assertRaisesRegex(ValueError,'release checks'):
                SkeletonRecognizer(path)
            with patch('mediapipe.solutions.hands.Hands'):
                model=SkeletonRecognizer(path,allow_demo=True)
                self.assertEqual(model.release_channel,'demo')
                self.assertEqual(model.experimental_labels,['H','K','R'])
                model.close()


if __name__=='__main__':unittest.main()
