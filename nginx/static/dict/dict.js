// ASR Support
//webkitURL is deprecated but nevertheless
const URL = window.URL || window.webkitURL;

const li = document.createElement('li');
var gumStream; //stream from getUserMedia()
var rec; //Recorder.js object
var input; //MediaStreamAudioSourceNode we'll be recording
var haveMic = false;
// shim for AudioContext when it's not avb.
const AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext; //audio context to help us record

const recordButton = document.getElementById('recordButton');
const stopButton = document.getElementById('stopButton');
const pauseButton = document.getElementById('pauseButton');

const asrUrl =
  'http://hw0-mke.tovera.com:19000/api/asr?task=transcribe&output=json&model=large';

//add events to those 2 buttons
recordButton.addEventListener('click', startRecording);
stopButton.addEventListener('click', stopRecording);
pauseButton.addEventListener('click', pauseRecording);

var langMap;

fetch('./langmap.json')
  .then((response) => response.json())
  .then((json) => (langMap = json));

console.log(JSON.stringify(langMap));

navigator.mediaDevices
  .getUserMedia({ audio: true })
  .then(function (stream) {
    console.log('You let me use your mic!');
  })
  .catch(function (err) {
    console.log('No mic for you!');
  });

function startRecording() {
  console.log('recordButton clicked');

  /*
		Simple constraints object, for more advanced audio features see
		https://addpipe.com/blog/audio-constraints-getusermedia/
	*/

  const constraints = { audio: true, video: false };

  /*
    	Disable the record button until we get a success or fail from getUserMedia() 
	*/

  recordButton.disabled = true;
  stopButton.disabled = false;
  pauseButton.disabled = false;

  /*
    	We're using the standard promise based getUserMedia() 
    	https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
	*/

  navigator.mediaDevices
    .getUserMedia(constraints)
    .then(function (stream) {
      console.log(
        'getUserMedia() success, stream created, initializing Recorder.js ...'
      );

      /*
			create an audio context after getUserMedia is called
			sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
			the sampleRate defaults to the one set in your OS for your playback device

		*/
      audioContext = new AudioContext();

      //update the format
      document.getElementById('formats').innerHTML =
        'Format: 1 channel pcm @ ' + audioContext.sampleRate / 1000 + 'kHz';

      /*  assign to gumStream for later use  */
      gumStream = stream;

      /* use the stream */
      input = audioContext.createMediaStreamSource(stream);

      /* 
			Create the Recorder object and configure to record mono sound (1 channel)
			Recording 2 channels  will double the file size
		*/
      rec = new Recorder(input, { numChannels: 1 });

      //start the recording process
      rec.record();

      console.log('Recording started');
    })
    .catch(function (err) {
      //enable the record button if getUserMedia() fails
      recordButton.disabled = false;
      stopButton.disabled = true;
      pauseButton.disabled = true;
    });
}

function pauseRecording() {
  console.log('pauseButton clicked rec.recording=', rec.recording);
  if (rec.recording) {
    //pause
    rec.stop();
    pauseButton.innerHTML = 'Resume';
  } else {
    //resume
    rec.record();
    pauseButton.innerHTML = 'Pause';
  }
}

function stopRecording() {
  console.log('stopButton clicked');

  //disable the stop button, enable the record too allow for new recordings
  stopButton.disabled = true;
  recordButton.disabled = false;
  pauseButton.disabled = true;

  //reset button just in case the recording is stopped while paused
  pauseButton.innerHTML = 'Pause';

  //tell the recorder to stop the recording
  rec.stop();

  //stop microphone access
  gumStream.getAudioTracks()[0].stop();

  //create the wav blob and pass it on to createDownloadLink
  rec.exportWAV(whisperUpload);
  rec.exportWAV(createDownloadLink);
}

function whisperUpload(blob) {
  if (haveMic) {
    const device = getDeviceWithSelectedId();
    device.setLed(0, 1);
  }
  const filename = new Date().toISOString();
  const xhr = new XMLHttpRequest();
  xhr.onload = function (e) {
    if (this.readyState === 4) {
      //console.log('Server returned: ', e.target.responseText);
      const jsonResponse = JSON.parse(xhr.responseText);
      const asrText = jsonResponse.text;
      const asrLang = jsonResponse.language;
      const asrTime = jsonResponse.infer_time;
      if (jsonResponse.translation) {
        var asrTranslation = jsonResponse.translation;
      }
      WriteLog(`AIR ASR done on language ${asrLang} in ${asrTime} ms`);
      WriteLog(`AIR ASR transcription: ${asrText}`);
      if (asrTranslation) {
        WriteLog(`AIR ASR translation: ${asrTranslation}`);
        var editorText = asrTranslation;
      } else {
        var editorText = asrText;
      }
      quill.focus();
      quill.setText(`${editorText}\n`);
      quill.setSelection(quill.getLength(), 0);
      console.log(JSON.stringify(asrText));
      li.appendChild(document.createTextNode(` ASR results: ${editorText}`));
      if (haveMic) {
        device.setLed(0, 0);
      }
    }
  };
  const fd = new FormData();
  fd.append('audio_file', blob, filename);
  xhr.open('POST', asrUrl, true);
  xhr.send(fd);
}

function createDownloadLink(blob) {
  const url = URL.createObjectURL(blob);
  const au = document.createElement('audio');
  const link = document.createElement('a');

  //name of .wav file to use during upload and download (without extendion)
  const filename = new Date().toISOString();

  //add controls to the <audio> element
  au.controls = true;
  au.src = url;

  //save to disk link
  link.href = url;
  link.download = filename + '.wav'; //download forces the browser to donwload the file using the  filename
  //link.innerHTML = 'Save to disk';

  //add the new audio element to li
  li.appendChild(au);

  //add the filename to the li
  //li.appendChild(document.createTextNode(filename + '.wav '));
  //li.appendChild(document.createTextNode(asrText));
  //add the save to disk link to li
  li.appendChild(link);

  //upload link
  const upload = document.createElement('a');
  upload.href = '#';
  //upload.innerHTML = 'Upload';
  upload.addEventListener('click', function (event) {
    const xhr = new XMLHttpRequest();
    xhr.onload = function (e) {
      if (this.readyState === 4) {
        //console.log('Server returned: ', e.target.responseText);
        var jsonResponse = JSON.parse(xhr.responseText);
        var asrText = jsonResponse.text;
        WriteLog(`ASR output: ${asrText}`);
        console.log(JSON.stringify(jsonResponse.text));
      }
    };
    const fd = new FormData();
    fd.append('audio_file', blob, filename);
    xhr.open('POST', asrUrl, true);
    xhr.send(fd);
  });
  li.appendChild(document.createTextNode(' ')); //add a space in between
  li.appendChild(upload); //add the upload link to li

  //add the li element to the ol
  recordingsList.appendChild(li);
}

// Dictation device lib

let deviceManager = null;

var lastButton;
function WriteLog(message) {
  log = document.getElementById('log');
  log.value = `${log.value}${new Date().toLocaleTimeString()}: ${message}\n`;
  log.scrollTop = log.scrollHeight;
}

/*
window.onerror = (error) => {
  WriteLog(`onError() \t ${error}`);
};

window.addEventListener('unhandledrejection', (promiseRejectionEvent) => {
  WriteLog(`onError() \t ${promiseRejectionEvent.reason}`);
});
*/
function deviceToString(device) {
  return JSON.stringify({
    id: device.id,
    type: DictationSupport.DeviceType[device.getDeviceType()],
  });
}

function devicesToString(devices) {
  return JSON.stringify(
    devices.map((device) => JSON.parse(deviceToString(device)))
  );
}

function setDeviceIdInUi(device) {
  document.getElementById('id').value = device.id;
}

function getFirstItemOfSet(set) {
  for (let item of set) {
    if (item) {
      return item;
    }
  }
  return undefined;
}

function onButtonEvent(device, bitMask) {
  const events = new Set();
  for (const value of Object.values(DictationSupport.ButtonEvent)) {
    const valueAsNumber = Number(value);
    if (isNaN(valueAsNumber)) continue;
    if (bitMask & valueAsNumber) {
      events.add(DictationSupport.ButtonEvent[valueAsNumber]);
    }
    //let logEvent = JSON.stringify(DictationSupport.ButtonEvent);
    //console.log(logEvent)
  }

  const button = getFirstItemOfSet(events);
  if (button) {
    lastButton = button;
    WriteLog(`${lastButton} button pressed`);
  } else {
    WriteLog(`${lastButton} button let go`);
  }

  if (button === 'RECORD') {
    console.log('RECORD PRESSED');

    device.setLed(0, 3);
    WriteLog('Start recording');
    startRecording();
  }
  console.log(`Last button is ${lastButton}`);
  if (lastButton === 'RECORD' && !button) {
    console.log('RECORD LET GO');
    device.setLed(0, 0);
    WriteLog('Stop recording');
    stopRecording();
  }
}

function onMotionEvent(device, motionEvent) {
  WriteLog(`Got motion ${DictationSupport.MotionEvent[motionEvent]}`);
}

function onDeviceConnected(device) {
  WriteLog(`onDeviceConnected() \t ${deviceToString(device)}`);
  setDeviceIdInUi(device);
}

function onDeviceDisconnected(device) {
  WriteLog(`onDeviceDisconnected() \t ${deviceToString(device)}`);
}

async function init() {
  if (deviceManager === null) {
    deviceManager = new DictationSupport.DictationDeviceManager();
    deviceManager.addButtonEventListener(onButtonEvent);
    deviceManager.addDeviceConnectedEventListener(onDeviceConnected);
    deviceManager.addDeviceDisconnectedEventListener(onDeviceDisconnected);
    deviceManager.addMotionEventListener(onMotionEvent);
  }
  await deviceManager.init();
  //WriteLog('init() done');
  getDevices();
}

async function shutdown() {
  await deviceManager.shutdown();
  WriteLog('shutdown() done');
}

async function requestDevice() {
  const devices = await deviceManager.requestDevice();
  WriteLog(`requestDevice() devices: ${devicesToString(devices)}`);
  if (devices.length !== 0) {
    setDeviceIdInUi(devices[0]);
  }
}

function getDevices() {
  const devices = deviceManager.getDevices();

  if (devices.length > 0) {
    setDeviceIdInUi(devices[0]);
    WriteLog(`Got dictation device ${devicesToString(devices)}`);
    haveMic = true;
  } else {
    WriteLog(`Failed to get dictation device - use click buttons`);
  }
}

function getDeviceWithSelectedId() {
  const devices = deviceManager.getDevices();
  const id = parseInt(document.getElementById('id').value);
  const device = devices.find((device) => device.id === id);
  if (device === undefined) {
    const errorMessage = `no device with ID ${id} `;
    WriteLog(errorMessage);
    throw new Error(errorMessage);
  }
  return device;
}

async function getEventMode() {
  const device = getDeviceWithSelectedId();
  const eventMode = await device.getEventMode();
  WriteLog(
    `getEventMode() \t device: ${deviceToString(device)}
} \t eventMode: ${DictationSupport.EventMode[eventMode]} `
  );
}

async function setEventMode() {
  const device = getDeviceWithSelectedId();
  const eventMode = parseInt(document.getElementById('eventMode').value);
  await device.setEventMode(eventMode);
  WriteLog(
    `setEventMode() \t device: ${deviceToString(
      device
    )}} \t eventMode: ${eventMode} `
  );
}

async function setSimpleLedState() {
  const device = getDeviceWithSelectedId();
  const simpleLedState = parseInt(
    document.getElementById('simpleLEDState').value
  );
  await device.setSimpleLedState(simpleLedState);
  WriteLog(
    `setSimpleLedState() \t device: ${deviceToString(
      device
    )} \t simpleLedState: ${simpleLedState} `
  );
}

async function setLed() {
  const device = getDeviceWithSelectedId();
  const index = parseInt(document.getElementById('ledIndex').value);
  const mode = parseInt(document.getElementById('ledMode').value);
  await device.setLed(index, mode);
  WriteLog(
    `setLed() \t device: ${deviceToString(
      device
    )} \t index: ${index} \t mode: ${mode} `
  );
}

async function setLedPM3() {
  const device = getDeviceWithSelectedId();
  const state = parseInt(document.getElementById('ledState').value);
  await device.setLed(state);
  WriteLog(`setLed() \t device: ${deviceToString(device)} \t state: ${state}`);
}
init();
