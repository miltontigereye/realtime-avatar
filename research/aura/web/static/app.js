'use strict';
const $ = (id) => document.getElementById(id);
const video = $('video');
const MAX_PENDING = 4;
const state = {
  active:false, ready:false, finishing:false, source:null, socket:null, session:null,
  sent:0, generated:0, paused:false, context:null, mic:null, worklet:null,
  mediaSource:null, sourceBuffer:null, mediaUrl:null, segments:[], ended:false,
  image:null, imageUrl:null, samplePlaying:false, error:false, mediaBytes:0,
  playStarted:false, audioFile:null, audioSamples:null, audioDecoding:false, audioSelection:0,
  flushing:false, flushResolve:null,
  lastVideoTime:0, advancedAt:0, recoveredAt:0,
};
const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));
const selectedModel = () => document.querySelector('input[name="model"]:checked').value;
const modelName = () => selectedModel() === 'aura-5b' ? '5B' : '1.3B';
const levels = Array(70).fill(0);
let micTest = null, micTestUrl = null;

function setStatus(text, kind = '') {
  $('stream-state').textContent = text;
  $('stream-dot').className = `dot ${kind}`;
}
function showError(message) {
  $('error').textContent = message;
  $('error').hidden = false;
  state.error = true;
  $('loading').hidden = true;
  setStatus('Stream needs attention');
}
function setControls(busy) {
  $('start').hidden = busy;
  $('sample').hidden = busy;
  $('start').disabled = state.audioDecoding || !!micTest;
  $('sample').disabled = state.audioDecoding || !!micTest;
  $('mic-test').disabled = busy || state.audioDecoding;
  $('audio-input').disabled = busy;
  $('audio-start').hidden = busy;
  $('audio-start').disabled = busy || !!micTest || state.audioDecoding || !state.audioSamples;
  $('stop-controls').hidden = !busy;
  $('finish').disabled = !state.ready || state.finishing;
  $('cancel').disabled = busy && !state.socket;
  $('portrait-input').disabled = busy;
  $('reset-portrait').disabled = busy;
  $('unload').disabled = busy;
  for (const radio of document.querySelectorAll('input[name="model"]')) radio.disabled = busy;
  $('compile').disabled = busy || selectedModel() === 'aura-5b';
}
function updateModel() {
  $('model-label').textContent = modelName();
  $('output-rate').textContent = selectedModel() === 'aura-5b' ? '25 fps playback' : '16 fps playback';
  $('compile').disabled = selectedModel() === 'aura-5b';
  if ($('compile').disabled) $('compile').checked = false;
  $('compile-help').textContent = selectedModel() === 'aura-5b'
    ? '5B uses automatic memory offloading on smaller GPUs. Compilation is disabled for this model.'
    : 'First-use compilation can take several minutes. Off by default.';
}
function setPortrait(file) {
  if (state.active) return;
  if (file && (!/^image\/(png|jpeg|webp)$/.test(file.type) || file.size > 10 * 1024 * 1024)) {
    showError('Choose a PNG, JPEG or WebP portrait smaller than 10 MB.');
    return;
  }
  if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
  state.image = file;
  state.imageUrl = file ? URL.createObjectURL(file) : null;
  const src = state.imageUrl || '/sample/portrait.jpg';
  $('portrait-preview').src = src;
  $('stage-portrait').src = src;
  $('portrait-name').textContent = file ? file.name : 'Sample portrait';
  $('error').hidden = true;
  video.pause();video.hidden = true;$('placeholder').hidden = false;$('resolution').hidden = true;
}
function bufferedAhead() {
  if (!state.sourceBuffer || !state.sourceBuffer.buffered.length) return 0;
  try {return Math.max(0, state.sourceBuffer.buffered.end(state.sourceBuffer.buffered.length - 1) - video.currentTime);}
  catch {return 0;}
}
function updateCapture() {
  const pending = Math.max(0, state.sent / 24000 - state.generated);
  const ahead = bufferedAhead();
  const paused = !state.active || !state.ready || state.finishing || (state.paused
    ? pending > 1 || ahead > 4 : pending >= MAX_PENDING || ahead >= 8);
  if (paused !== state.paused) {
    state.paused = paused;
    state.worklet?.port.postMessage({enabled:!paused});
    if (state.mic) state.mic.getAudioTracks().forEach(track => {track.enabled = !paused;});
  }
  if (state.active && state.ready && !state.finishing) {
    if (paused) {
      $('capture-label').textContent = 'Capture paused';
      setStatus('Catching up', 'busy');
      $('session-message').textContent = ahead >= 8
        ? 'Capture is paused until playback catches up. Press play if the video is paused.'
        : state.source === 'microphone' ? 'Capture is paused while queued audio renders. It resumes automatically; speak when the microphone is listening.' : 'Audio file input is paused while queued audio renders. It resumes automatically without skipping any audio.';
    } else {
      $('capture-label').textContent = state.source === 'microphone' ? 'Microphone live' : 'Audio input streaming';
      setStatus(state.source === 'microphone' ? 'Listening' : 'Streaming audio', 'live');
      $('session-message').textContent = 'Finish stream keeps queued audio and completes the video. Cancel stops after the current chunk.';
    }
  }
  $('queue-value').innerHTML = `${pending.toFixed(1)} <i>sec</i>`;
  $('queue-fill').style.width = `${Math.min(100, pending / MAX_PENDING * 100)}%`;
  const seconds = Math.floor(state.sent / 24000);
  $('recording-time').textContent = `${String(Math.floor(seconds / 60)).padStart(2,'0')}:${String(seconds % 60).padStart(2,'0')}`;
}
function sendPCM(samples, force = false) {
  updateCapture();
  if (!state.active || !state.ready || (!force && (state.finishing || state.paused)) || state.socket?.readyState !== WebSocket.OPEN) return false;
  if (state.socket.bufferedAmount > 1024 * 1024) {
    cancelStream('The connection cannot keep up with audio. Try again locally.');
    return false;
  }
  state.socket.send(samples.buffer.slice(samples.byteOffset, samples.byteOffset + samples.byteLength));
  state.sent += samples.length;
  let sum = 0;
  for (const sample of samples) sum += sample * sample;
  levels.shift();levels.push(Math.min(1, Math.sqrt(sum / samples.length) * 7));
  updateCapture();
  return true;
}
async function stopAudio() {
  state.worklet?.port.postMessage({enabled:false});
  state.worklet?.disconnect();state.worklet = null;
  state.mic?.getTracks().forEach(track => track.stop());state.mic = null;
  if (state.context) {await state.context.close().catch(()=>{});state.context = null;}
  $('capture-label').textContent = 'Audio input stopped';
}
async function preparePlayback() {
  const mime = 'video/mp4; codecs="avc1.42C01F,mp4a.40.2"';
  if (!window.MediaSource || !MediaSource.isTypeSupported(mime)) {
    throw new Error('This browser does not support the live video format. Open the demo in Chrome or Edge.');
  }
  if (state.mediaUrl) URL.revokeObjectURL(state.mediaUrl);
  state.mediaSource = new MediaSource();
  state.mediaUrl = URL.createObjectURL(state.mediaSource);
  video.src = state.mediaUrl;
  state.segments = [];state.ended = false;state.mediaBytes = 0;
  state.playStarted = false;
  state.lastVideoTime = 0;state.advancedAt = performance.now();state.recoveredAt = 0;
  await new Promise((resolve,reject)=>{
    const timeout = setTimeout(()=>reject(new Error('Could not initialize the video player.')),10000);
    state.mediaSource.addEventListener('sourceopen',()=>{
      clearTimeout(timeout);
      try {
        state.sourceBuffer = state.mediaSource.addSourceBuffer(mime);
        state.sourceBuffer.addEventListener('updateend',()=>{
          flushSegments();
          if (video.buffered.length && !state.error) {
            $('placeholder').hidden = true;video.hidden = false;
            if (!state.playStarted) {
              state.playStarted = true;
              video.play().catch(()=>{$('play').hidden = false;});
            }
          }
        });
        state.sourceBuffer.addEventListener('error',()=>cancelStream('The browser could not decode the live stream.'));
        resolve();
      } catch(error) {reject(error);}
    },{once:true});
  });
}
function flushSegments() {
  const source = state.sourceBuffer;
  if (!source || source.updating || state.mediaSource.readyState !== 'open') return;
  try {
    if (video.currentTime > 25 && source.buffered.length && source.buffered.start(0) < video.currentTime - 20) {
      source.remove(0, video.currentTime - 20);
      return;
    }
    if (state.segments.length) source.appendBuffer(state.segments.shift());
    else if (state.ended) state.mediaSource.endOfStream();
  } catch(error) {cancelStream(`Video playback failed: ${error.message}`);}
}
function recoverPlayback() {
  const now = performance.now();
  if (video.currentTime > state.lastVideoTime + .02) {
    state.lastVideoTime = video.currentTime;state.advancedAt = now;
  }
  // Chromium can keep its audio renderer in rebuffering after an underrun,
  // even with a full new chunk buffered. A tiny in-buffer seek restarts it
  // without waiting for several more seconds of generated audio.
  if (state.playStarted && !video.paused && !video.seeking && video.readyState < 3 &&
      bufferedAhead() > .25 && now - state.advancedAt > 800 && now - state.recoveredAt > 1200) {
    state.recoveredAt = now;
    video.currentTime += .001;
    video.play().catch(()=>{$('play').hidden = false;});
  }
}
async function stopMicTest(test = micTest) {
  if (!test || test !== micTest || test.stopping) return;
  test.stopping = true;
  clearInterval(test.meterTimer);clearTimeout(test.timer);
  if (test.recorder?.state === 'recording') {
    await new Promise(resolve=>{
      test.recorder.addEventListener('stop',resolve,{once:true});test.recorder.stop();
    });
  }
  test.stream?.getTracks().forEach(track=>track.stop());
  await test.context?.close().catch(()=>{});
  if (test.chunks.length) {
    if (micTestUrl) URL.revokeObjectURL(micTestUrl);
    micTestUrl = URL.createObjectURL(new Blob(test.chunks,{type:test.recorder.mimeType}));
    $('mic-test-playback').src = micTestUrl;$('mic-test-playback').hidden = false;
    $('mic-test-status').textContent = test.peak > .005
      ? 'Sound detected. Press play below to listen to your recording.'
      : 'Very little sound detected. Check your microphone and input volume, then try again.';
  } else if (!test.failed) $('mic-test-status').textContent = 'Test stopped. You can try again.';
  $('mic-test-level').value = 0;$('mic-test-level').hidden = true;
  micTest = null;$('mic-test').textContent = 'Test microphone';setControls(state.active);
}
async function toggleMicTest() {
  if (micTest) {await stopMicTest();return;}
  if (state.active || state.audioDecoding) return;
  const test = {stream:null,context:null,recorder:null,chunks:[],peak:0,stopping:false};
  micTest = test;setControls(false);video.pause();
  $('mic-test').textContent = 'Stop microphone test';
  $('mic-test-status').textContent = 'Allow microphone access to begin the test…';
  $('mic-test-playback').pause();$('mic-test-playback').hidden = true;
  try {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      throw new Error('Microphone testing requires localhost or HTTPS in Chrome or Edge.');
    }
    const stream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true}});
    if (test !== micTest || test.stopping) {stream.getTracks().forEach(track=>track.stop());return;}
    test.stream = stream;test.context = new AudioContext();await test.context.resume();
    if (test !== micTest || test.stopping) return;
    const analyser = test.context.createAnalyser();analyser.fftSize = 2048;
    test.context.createMediaStreamSource(stream).connect(analyser);
    test.recorder = new MediaRecorder(stream);
    test.recorder.addEventListener('dataavailable',event=>{if(event.data.size)test.chunks.push(event.data);});
    test.recorder.addEventListener('error',()=>{
      test.failed = true;$('mic-test-status').textContent = 'Recording failed. Check your microphone and try again.';stopMicTest(test);
    });
    test.recorder.start();
    const started = performance.now(), samples = new Float32Array(analyser.fftSize);
    $('mic-test-level').hidden = false;
    test.meterTimer = setInterval(()=>{
      analyser.getFloatTimeDomainData(samples);
      let sum = 0;for (const sample of samples) sum += sample * sample;
      const rms = Math.sqrt(sum / samples.length);test.peak = Math.max(test.peak,rms);
      $('mic-test-level').value = Math.min(1,rms * 7);
      $('mic-test-status').textContent = `Recording microphone · ${Math.min(5,(performance.now()-started)/1000).toFixed(1)} / 5 seconds`;
    },100);
    test.timer = setTimeout(()=>stopMicTest(test),5000);
  } catch(error) {
    if (test !== micTest || test.stopping) return;
    test.failed = true;
    $('mic-test-status').textContent = error.name === 'NotAllowedError'
      ? 'Microphone access was denied. Allow it in your browser’s site settings and try again.'
      : error.name === 'NotFoundError' ? 'No microphone found. Connect one and try again.' : error.message;
    await stopMicTest(test);
  }
}
async function prepareMicrophone() {
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
    throw new Error('Microphone capture requires localhost or HTTPS and a browser with AudioWorklet support.');
  }
  state.context = new AudioContext({sampleRate:24000});
  await state.context.resume();
  const mic = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
  if (!state.active) {mic.getTracks().forEach(track=>track.stop());return;}
  state.mic = mic;
  mic.getTracks().forEach(track=>{track.enabled = false;});
  await state.context.audioWorklet.addModule('/static/mic-worklet.js');
  state.worklet = new AudioWorkletNode(state.context,'aura-microphone');
  state.worklet.port.onmessage = ({data})=>{
    if (data.flushed) state.flushResolve?.();
    else sendPCM(data,state.flushing);
  };
  const input = state.context.createMediaStreamSource(mic);
  const silent = state.context.createGain();silent.gain.value = 0;
  input.connect(state.worklet).connect(silent).connect(state.context.destination);
}
async function decodeAudio(bytes) {
  const decoding = new AudioContext({sampleRate:24000});
  try {
    const audio = await decoding.decodeAudioData(bytes);
    if (!audio.length) throw new Error('The audio file is empty.');
    const samples = new Float32Array(audio.length);
    for (let channel = 0; channel < audio.numberOfChannels; channel++) {
      const data = audio.getChannelData(channel);
      for (let i = 0; i < data.length; i++) samples[i] += data[i] / audio.numberOfChannels;
    }
    return samples;
  } finally {await decoding.close();}
}
async function selectAudio(file) {
  if (state.active) return;
  const selection = ++state.audioSelection;
  state.audioFile = null;state.audioSamples = null;
  state.audioDecoding = !!file;setControls(false);
  if (!file) {state.audioDecoding = false;$('audio-name').textContent = 'Choose speech audio to stream.';return;}
  $('audio-name').textContent = 'Reading audio…';
  try {
    if (file.size > 50 * 1024 * 1024) throw new Error('Choose an audio file smaller than 50 MB.');
    const samples = await decodeAudio(await file.arrayBuffer());
    if (selection !== state.audioSelection) return;
    state.audioSamples = samples;state.audioFile = file;
    $('audio-name').textContent = `${file.name} · ${(samples.length / 24000).toFixed(1)} seconds`;
    if (!state.active) {
      $('error').hidden = true;state.error = false;
      setStatus('Ready to create');
    }
  } catch(error) {
    if (selection !== state.audioSelection) return;
    $('audio-name').textContent = 'Could not read this file. Try WAV or MP3.';
    showError(error.message.startsWith('Choose') ? error.message : 'This audio file could not be decoded. Try a WAV or MP3 file.');
  } finally {
    if (selection === state.audioSelection) {state.audioDecoding = false;setControls(state.active);}
  }
}
async function runAudio() {
  state.samplePlaying = true;
  const samples = state.source === 'file' ? state.audioSamples
    : await decodeAudio(await (await fetch('/sample/speech.wav')).arrayBuffer());
  let position = 0;
  while (state.active && !state.finishing && position < samples.length) {
    const next = samples.subarray(position,Math.min(position+2048,samples.length));
    if (sendPCM(next)) {position += next.length;await wait(next.length / 24);}
    else await wait(50);
  }
  state.samplePlaying = false;
  if (state.active && !state.finishing) finishStream();
}
async function startStream(source) {
  if (state.active || micTest || (source === 'file' && !state.audioSamples)) return;
  $('mic-test-playback').pause();
  state.active = true;state.ready = false;state.finishing = false;state.error = false;
  state.source = source;state.sent = 0;state.generated = 0;state.paused = true;state.session = null;state.socket = null;
  $('error').hidden = true;$('download').hidden = true;$('play').hidden = true;
  $('generation-rate').innerHTML = '— <i>fps</i>';$('generated-value').innerHTML = '0.0 <i>sec</i>';
  $('chunk-value').textContent = 'Waiting for your voice';
  $('loading-message').textContent = source === 'microphone' ? 'Connecting your microphone…' : 'Preparing your stream…';
  $('loading').hidden = false;
  setStatus('Preparing stream', 'busy');setControls(true);updateCapture();
  try {
    if (source === 'microphone') await prepareMicrophone();
    if (!state.active) return;
    await preparePlayback();
    const form = new FormData();
    form.append('model',selectedModel());form.append('compile',String($('compile').checked));
    if (state.image) form.append('image',state.image,state.image.name);
    const response = await fetch('/api/sessions',{method:'POST',body:form});
    if (!response.ok) throw new Error(await response.text());
    const session = await response.json();state.session = session.id;
    if (!state.active) {await fetch(`/api/sessions/${session.id}`,{method:'DELETE'});return;}
    const url = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}${session.websocket}`;
    const socket = new WebSocket(url);state.socket = socket;socket.binaryType = 'arraybuffer';
    socket.onopen = ()=>{
      $('cancel').disabled = false;
      if (!state.active || state.finishing) socket.send(JSON.stringify({type:'cancel'}));
    };
    socket.onmessage = ({data})=>{
      if (data instanceof ArrayBuffer) {
        state.mediaBytes += data.byteLength;state.segments.push(data);flushSegments();
        return;
      }
      const message = JSON.parse(data);
      if (message.type === 'status') {
        const text = message.state === 'loading' ? `Loading ${modelName()}…` : message.message;
        $('loading-message').textContent = text;setStatus(text,'busy');
      } else if (message.type === 'ready') {
        state.ready = true;state.paused = true;
        $('loading').hidden = true;$('finish').disabled = false;
        $('resolution').textContent = `${message.width} × ${message.height} · ${message.fps} fps`;
        $('resolution').hidden = false;
        updateCapture();
        if (source !== 'microphone') runAudio().catch(error=>cancelStream(error.message));
      } else if (message.type === 'buffer') {
        state.generated = Math.max(state.generated,message.generated_seconds);updateCapture();
      } else if (message.type === 'chunk') {
        state.generated = message.generated_seconds;
        $('generation-rate').innerHTML = `${message.generation_fps.toFixed(1)} <i>fps</i>`;
        $('generated-value').innerHTML = `${message.generated_seconds.toFixed(1)} <i>sec</i>`;
        $('chunk-value').textContent = `${message.chunk} ${message.chunk === 1 ? 'chunk' : 'chunks'} delivered`;
        updateCapture();
      } else if (message.type === 'done') {
        state.ended = true;flushSegments();
        $('download').href = message.download;$('download').setAttribute('download','');$('download').hidden = state.sent === 0;
        setStatus(state.sent ? 'Stream complete' : 'No audio captured','ready');
        $('session-message').textContent = state.sent ? 'Your stream is complete. Play it again or download the video.' : 'No audio was captured. Start a new stream to try again.';
      } else if (message.type === 'error') {
        showError(message.message);state.finishing = true;stopAudio();
      }
    };
    socket.onerror = ()=>showError('The connection to the local demo was interrupted. Check that the server is running.');
    socket.onclose = ()=>{
      state.active = false;state.ready = false;stopAudio();$('loading').hidden = true;setControls(false);
      if (!state.ended && !state.error) {setStatus('Stream stopped');$('session-message').textContent = 'Stream cancelled. You can start a new session.';}
      refreshConnection();
    };
  } catch(error) {
    state.active = false;state.ready = false;await stopAudio();setControls(false);showError(error.message);
    if (state.session) await fetch(`/api/sessions/${state.session}`,{method:'DELETE'}).catch(()=>{});
  }
}
async function finishStream() {
  if (!state.active || !state.ready || state.finishing) return;
  state.finishing = true;$('finish').disabled = true;
  if (state.worklet) {
    state.flushing = true;
    await new Promise(resolve=>{
      state.flushResolve = resolve;
      state.worklet.port.postMessage({flush:true});
      setTimeout(resolve,200);
    });
    state.flushing = false;state.flushResolve = null;
  }
  await stopAudio();
  setStatus('Finishing queued audio','busy');$('session-message').textContent = 'Capture has stopped. Finishing the remaining audio and video…';
  if (state.socket?.readyState === WebSocket.OPEN) state.socket.send(JSON.stringify({type:'finish'}));
}
function cancelStream(message) {
  state.finishing = true;stopAudio();$('finish').disabled = true;$('cancel').disabled = true;
  if (message) showError(message);
  else {setStatus('Stopping stream','busy');$('session-message').textContent = 'Stopping after the current generation chunk…';}
  if (state.socket?.readyState === WebSocket.OPEN) state.socket.send(JSON.stringify({type:'cancel'}));
  else if (state.socket?.readyState !== WebSocket.CONNECTING) {state.active = false;setControls(false);$('loading').hidden = true;}
}
async function refreshConnection() {
  try {
    const response = await fetch('/api/config');if (!response.ok) throw new Error();
    const config = await response.json();
    $('connection').textContent = config.gpu;
    $('connection-dot').className = `dot ${config.busy ? 'busy' : 'ready'}`;
  } catch {$('connection').textContent = 'Local server unavailable';$('connection-dot').className = 'dot';}
}
$('portrait-input').addEventListener('change',event=>setPortrait(event.target.files[0]));
$('dropzone').tabIndex = 0;
$('dropzone').setAttribute('role','button');
$('dropzone').setAttribute('aria-label','Upload target portrait');
$('dropzone').addEventListener('keydown',event=>{
  if (event.key === 'Enter' || event.key === ' ') {event.preventDefault();if(!state.active)$('portrait-input').click();}
});
$('reset-portrait').addEventListener('click',()=>{$('portrait-input').value='';setPortrait(null);});
$('dropzone').addEventListener('dragover',event=>{event.preventDefault();$('dropzone').classList.add('drag');});
$('dropzone').addEventListener('dragleave',()=>$('dropzone').classList.remove('drag'));
$('dropzone').addEventListener('drop',event=>{event.preventDefault();$('dropzone').classList.remove('drag');setPortrait(event.dataTransfer.files[0]);});
for (const radio of document.querySelectorAll('input[name="model"]')) radio.addEventListener('change',updateModel);
$('mic-test').addEventListener('click',toggleMicTest);
$('start').addEventListener('click',()=>startStream('microphone'));
$('sample').addEventListener('click',()=>startStream('sample'));
$('audio-input').addEventListener('change',event=>selectAudio(event.target.files[0]));
$('audio-start').addEventListener('click',()=>startStream('file'));
$('finish').addEventListener('click',finishStream);$('cancel').addEventListener('click',()=>cancelStream());
$('play').addEventListener('click',()=>video.play().then(()=>{$('play').hidden=true;}).catch(error=>showError(error.message)));
$('unload').addEventListener('click',async()=>{
  $('unload').disabled = true;
  try {const response = await fetch('/api/unload',{method:'POST'});if(!response.ok)throw new Error(await response.text());$('session-message').textContent='GPU memory released. The next stream will load its selected model.';await refreshConnection();}
  catch(error){showError(error.message);}finally{$('unload').disabled=false;}
});
window.addEventListener('beforeunload',()=>{micTest?.stream?.getTracks().forEach(track=>track.stop());state.socket?.close();state.mic?.getTracks().forEach(track=>track.stop());});
video.addEventListener('error',()=>{if(state.active)cancelStream('The video player could not read the generated stream.');});
setInterval(()=>{updateCapture();recoverPlayback();},200);setInterval(refreshConnection,10000);
function drawWaveform() {
  const context = $('waveform').getContext('2d'), width = 960, height = 72;
  context.clearRect(0,0,width,height);
  context.fillStyle = state.active && state.ready && !state.finishing && !state.paused ? '#70ebcd' : '#50616c';
  levels.forEach((level,i)=>{const size=3+level*56;context.fillRect(i*width/levels.length+2,(height-size)/2,4,size);});
  requestAnimationFrame(drawWaveform);
}
refreshConnection();updateModel();drawWaveform();
