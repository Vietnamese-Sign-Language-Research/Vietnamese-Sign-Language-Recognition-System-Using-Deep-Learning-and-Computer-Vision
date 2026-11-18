# Signing guide sources

## Letter illustrations and tutorials

The 23 PNGs in `letters/` are individual grayscale crops from
[Vietnam's national fingerspelling chart](https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2020/6/31603/31697-1-2020671-67217-2020-tt-bgddt.pdf#page=6),
section II, gazette page 51, published with Ministry of Education and Training
Circular 17/2020/TT-BGDĐT. The app includes only the letters its current
alphabet checkpoint recognizes. The drawings are educational references,
not photos from the training set.

Individual tutorial links open the corresponding letter of the
[Vietnamese Sign Language Dictionary](https://tudienngonngukyhieu.com/bang-chu-cai-ngon-ngu-ky-hieu).
K, L, S, and Đ link to the Hanoi Sign Language Training Center's
[full alphabet lesson](https://www.youtube.com/watch?v=ggQY-g4aQp8)
on YouTube. Tutorials may show regional variants; no letter videos are bundled.

## Sentence demonstration videos

These public videos are embedded using YouTube’s player. No source video is copied or redistributed. The demonstrations are learning references; they are not confirmed recordings from the model’s training set. Regional signing may differ.

Verified on 5 October 2026:

| Model label | Source | Video |
| --- | --- | --- |
| Xin chào. | [TokyoLife’s official lesson page](https://tokyolife.vn/blog/ngoi-nha-thien-than/cung-hoc-ngon-ngu-ky-hieu-voi-thien-than-tokyolife), lesson 1. Includes Xin chào, Cảm ơn and Xin lỗi. | [TokyoLife Channel](https://www.youtube.com/watch?v=4avGnxaLmb0) |
| Cảm ơn. | [Hanoi Sign Language Talking Dictionary, entry 104](https://talkingdictionary.swarthmore.edu/vietnamese_sign/?entry=104), Swarthmore College. Signer: Isaac Trương. | [Demonstration](https://www.youtube.com/watch?v=E5I_GHsdXAY) |
| Bạn tên là gì? | [Hanoi Sign Language Talking Dictionary, entry 109](https://talkingdictionary.swarthmore.edu/vietnamese_sign/?entry=109), Swarthmore College. Signer: Isaac Trương. | [Demonstration](https://www.youtube.com/watch?v=7kHmw59FUxI) |

The [original sentence dataset repository](https://github.com/khooinguyeen/Vietnamese-Sign-Language-Translation) publishes extracted landmark arrays rather than labeled recordings for all 60 supported phrases. Unverified matches are omitted from the catalog; the UI shows an unavailable state for them.

## Adding a verified demonstration

Add a record in `sentence-videos.json` with an exact model label, source attribution and either `provider: "youtube"` with its 11-character `video_id`, or `provider: "local"` with an owned/licensed MP4 under `/assets/signs/`. Local URLs must use lowercase letters, digits, slashes and hyphens. Optional YouTube `start` and `end` values are whole seconds. Only add timestamps checked against the actual video.

The shared interface loads the catalog independently of recognition, lists available videos first, and removes the player on close or navigation. Selecting a demonstration does not constrain or influence the model’s prediction.
