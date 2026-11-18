// Exercise the real client capture functions without opening a physical camera.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
};
const ok = data => ({ok:true, json:async () => data});
const tick = () => new Promise(resolve => setImmediate(resolve));

function client(fetch) {
  const elements = new Map();
  function element(id) {
    if (elements.has(id)) return elements.get(id);
    const span = {textContent:''};
    const value = {hidden:false, disabled:false, style:{}, textContent:'', dataset:{},
      videoWidth:640, videoHeight:480, readyState:2,
      classList:{toggle(){},add(){},remove(){}},
      addEventListener(){}, setAttribute(){}, removeAttribute(){}, pause(){}, load(){}, replaceChildren(){}, append(){},
      querySelector:selector => selector === 'span' ? span : {setAttribute(){}}};
    elements.set(id,value);return value;
  }
  const modes = ['letters','words','sentences'].map(mode => ({...element(mode),dataset:{mode}}));
  const context=vm.createContext({
    fetch, Blob, FormData, AbortController, DOMException, setTimeout, clearTimeout, performance,
    navigator:{}, window:{addEventListener(){}},
    document:{body:{dataset:{page:'webcam'}},getElementById:element,addEventListener(){},
      querySelectorAll:() => modes,
      createElement:() => ({getContext:() => ({drawImage(){}}),toBlob:callback => callback(new Blob(['frame'],{type:'image/jpeg'}))})}
  });
  const source=fs.readFileSync(path.join(__dirname,'../assets/studio.js'),'utf8').replace(/initialize\(\);\s*$/,'');
  vm.runInContext(source,context);
  vm.runInContext("state.models=[{id:'letters',available:true}]; state.stream={getTracks:()=>[{stop(){}}]};",context);
  return {run:source => vm.runInContext(source,context),element,modes};
}

test('Finish waits for the last frame, disables repeat clicks, then predicts exactly once',async () => {
  const frame=deferred(), calls=[];
  const app=client(async (url,options={}) => {
    calls.push(url);
    if (url==='/api/live/session') return ok({id:'capture',max_seconds:45});
    if (url.endsWith('/frame')) return frame.promise;
    if (url.endsWith('/finish')) return ok({phase:'complete',result:{status:'recognized',label:'C',message:'Sign recognized.'}});
    return ok({closed:true});
  });
  await app.run('startSign()');await tick();
  assert.equal(app.element('camera-button').disabled,false);
  assert.equal(app.element('camera-button').querySelector('span').textContent,'Finish sign');
  const finish=app.run('finishSign()');
  await app.run('finishSign()');
  assert.equal(calls.filter(url => url.endsWith('/finish')).length,0);
  assert.equal(app.element('camera-button').disabled,true);
  frame.resolve(ok({phase:'capturing',elapsed:12,hands:1}));
  await finish;
  assert.deepEqual(calls,['/api/live/session','/api/live/capture/frame','/api/live/capture/finish','/api/live/capture']);
  assert.equal(app.element('result-label').textContent,'C');
  assert.equal(app.element('camera-button').querySelector('span').textContent,'Start another sign');
  assert.equal(app.element('camera-button').disabled,false);
});

test('No match leaves the camera ready for another successful sign, even if cleanup stalls',async () => {
  let attempt=0;
  const app=client(async (url,options={}) => {
    if (url==='/api/live/session') return ok({id:'retry-'+(++attempt),max_seconds:45});
    if (url.endsWith('/frame')) return ok({phase:'waiting_hands',elapsed:1,hands:0});
    if (url.endsWith('/finish')) return ok({phase:'complete',result:attempt===1
      ? {status:'no_sign',label:null,message:'No supported sign found. Try again.'}
      : {status:'recognized',label:'B',message:'Sign recognized.'}});
    if (options.method==='DELETE') return new Promise(()=>{});
  });
  await app.run('startSign()');await app.run('finishSign()');
  assert.equal(app.element('result-status').textContent,'NO CLEAR MATCH');
  assert.equal(app.element('copy-result').disabled,true);
  assert.equal(app.element('camera-button').disabled,false);
  assert.equal(app.run('state.busy || state.finishing'),false);
  await app.run('startSign()');await app.run('finishSign()');
  assert.equal(app.element('result-label').textContent,'B');
  assert.equal(app.element('camera-button').disabled,false);
});

for (const failure of ['disconnect','503','malformed']) {
  test(`A ${failure} prediction response resets the spinner and allows a retry`,async () => {
    const app=client(async url => {
      if (url==='/api/live/session') return ok({id:'error',max_seconds:45});
      if (url.endsWith('/frame')) return ok({phase:'capturing',elapsed:1,hands:1});
      if (url.endsWith('/finish')) {
        if (failure==='disconnect') throw new TypeError('Failed to fetch');
        if (failure==='503') return {ok:false,json:async()=>{throw new Error('HTML instead of JSON');}};
        return ok({phase:'complete'});
      }
      return ok({closed:true});
    });
    await app.run('startSign()');await app.run('finishSign()');
    assert.equal(app.element('result-loading').hidden,true);
    assert.equal(app.element('camera-button').disabled,false);
    assert.equal(app.element('camera-button').querySelector('span').textContent,'Start another sign');
    assert.match(app.element('input-error').textContent,/try your sign again/i);
    assert.equal(app.run('state.busy || state.finishing || state.running'),false);
  });
}

test('Capturing beyond six seconds keeps Finish enabled and never requests a prediction',async () => {
  const calls=[];
  const app=client(async url => {
    calls.push(url);
    if (url==='/api/live/session') return ok({id:'long',max_seconds:45});
    if (url.endsWith('/frame')) return ok({phase:'capturing',elapsed:20,hands:1});
    return ok({closed:true});
  });
  await app.run('startSign()');await app.run('state.framePromise');
  assert.equal(app.run('state.running'),true);
  assert.equal(app.element('camera-button').disabled,false);
  assert.equal(app.element('capture-duration').textContent,'20.0 / 45s');
  assert.equal(calls.some(url=>url.endsWith('/finish')),false);
  await app.run('stopCamera()');
});

test('Turning off the camera cancels a pending final frame without predicting',async () => {
  const frame=deferred(),calls=[];
  const app=client(async (url,options={}) => {
    calls.push(url);
    if (url==='/api/live/session') return ok({id:'cancel',max_seconds:45});
    if (url.endsWith('/frame')) {
      options.signal.addEventListener('abort',()=>frame.reject(new DOMException('Cancelled','AbortError')));
      return frame.promise;
    }
    return ok({closed:true});
  });
  await app.run('startSign()');await tick();
  const finish=app.run('finishSign()');
  await app.run('stopCamera()');await finish;
  assert.equal(calls.some(url=>url.endsWith('/finish')),false);
  assert.equal(app.run('state.stream'),null);
  assert.equal(app.element('camera-button').querySelector('span').textContent,'Enable camera');
});

test('Cancelling while a session opens cannot restart capture or leak the late session',async () => {
  const creation=deferred(),calls=[];
  const app=client(async url => {
    calls.push(url);
    return url==='/api/live/session' ? creation.promise : ok({closed:true});
  });
  const start=app.run('startSign()');
  await app.run('stopCamera()');
  creation.resolve(ok({id:'late',max_seconds:45}));await start;
  assert.equal(app.run('state.running'),false);
  assert.equal(app.run('state.session'),null);
  assert.deepEqual(calls,['/api/live/session','/api/live/late']);
  assert.equal(app.element('camera-button').querySelector('span').textContent,'Enable camera');
});
