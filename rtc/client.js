// get DOM elements
var dataChannelLog = document.getElementById('data-channel'),
    iceConnectionLog = document.getElementById('ice-connection-state'),
    iceGatheringLog = document.getElementById('ice-gathering-state'),
    signalingLog = document.getElementById('signaling-state');

// peer connection
var pc = null;

// data channel
var dc = null, dcInterval = null;

var stop_time = null

var constraints = {
    audio: true,
    video: false
};

// track
var asr_track = null, asr_stream = null, asr_sender = null

document.addEventListener('DOMContentLoaded', function() {
    init()
 }, false);

function createPeerConnection() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    config.iceServers = [{urls: ['stun:stun.l.google.com:19302']}];

    pc = new RTCPeerConnection(config);

    // register some listeners to help debugging
    pc.addEventListener('icegatheringstatechange', function() {
        iceGatheringLog.textContent += ' -> ' + pc.iceGatheringState;
    }, false);
    iceGatheringLog.textContent = pc.iceGatheringState;

    pc.addEventListener('iceconnectionstatechange', function() {
        iceConnectionLog.textContent += ' -> ' + pc.iceConnectionState;
    }, false);
    iceConnectionLog.textContent = pc.iceConnectionState;

    pc.addEventListener('signalingstatechange', function() {
        signalingLog.textContent += ' -> ' + pc.signalingState;
    }, false);
    signalingLog.textContent = pc.signalingState;

    // connect audio
    pc.addEventListener('track', function(evt) {
        document.getElementById('audio').srcObject = evt.streams[0];
    });

    return pc;
}

function negotiate() {
    return pc.createOffer().then(function(offer) {
        return pc.setLocalDescription(offer);
    }).then(function() {
        // wait for ICE gathering to complete
        return new Promise(function(resolve) {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                function checkState() {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(function() {
        var offer = pc.localDescription;
        var codec;

        console.log(JSON.stringify(offer.sdp))
        codec = 'opus/48000/2'
        if (codec !== 'default') {
            offer.sdp = sdpFilterCodec('audio', codec, offer.sdp);
        }

        // The route in FastAPI supports all of the usual URL params to control ASR
        return fetch('/api/rtc/asr?model=large', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then(function(response) {
        return response.json();
    }).then(function(answer) {
        return pc.setRemoteDescription(answer);
    }).catch(function(e) {
        alert(e);
    });
}

function switchTrack(switch_track) {
    let current_track = pc.getSenders()[0]
    console.log("SWITCHING TRACK FROM")
    console.log(current_track)
    console.log("SWITCHING TRACK TO")
    console.log(switch_track)
    asr_sender.replaceTrack(switch_track);
}

function muteMic (mute) {
    if (mute) {
        console.log("Muting microphone")
    } else {
        console.log("Unmuting microphone")
    }
    asr_stream.getAudioTracks()[0].enabled = !mute;
};

function init() {
    pc = createPeerConnection();

    // Init DC
    var parameters = {'ordered': true}
    dc = pc.createDataChannel('chat', parameters);
    dc.onclose = function() {
        clearInterval(dcInterval);
        dataChannelLog.textContent += 'Disconnected from AIR ASR Service\n';
    };
    dc.onopen = function() {
        dataChannelLog.textContent += 'Connected to AIR ASR Service\n';
    };
    dc.onmessage = function(evt) {
        dataChannelLog.textContent += evt.data + '\n';
        let data = evt.data
        if (data.includes('Infer')) {
            const end = Date.now();
            let time_log = `Total time: ${end - stop_time} ms`
            dataChannelLog.textContent += time_log + '\n';
        }

        if (evt.data.substring(0, 4) === 'pong') {
            var elapsed_ms = current_stamp() - parseInt(evt.data.substring(5), 10);
            dataChannelLog.textContent += ' RTT ' + elapsed_ms + ' ms\n';
        }
    };

    if (constraints.audio) {
        navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
            stream.getTracks().forEach(function(track) {
                pc.addTrack(track, stream);
                asr_track = track
                asr_stream = stream
                asr_sender = pc.getSenders()[0]
                console.log('INIT - ASR TRACK')
                console.log(asr_track)
                console.log('INIT - ASR STREAM')
                console.log(asr_stream)
                console.log('INIT - ASR SENDER')
                console.log(asr_sender)

            });
            // After we init and negotiate replace track until we click start
            //muteMic(true)
            switchTrack(null)
            return negotiate();
        }, function(err) {
            alert('Could not acquire media: ' + err);
        });
    } else {
        negotiate();
    }
}

function stop() {
    // close local audio
    console.log('STOP')
    stop_time = Date.now()
    dc.send("stop");
    switchTrack(null)
    //muteMic(true)
}

function start() {
    console.log('START')
    //muteMic(false)
    switchTrack(asr_track)
    dc.send("start");
}

function disconnect() {
    // close data channel

    pc.getSenders().forEach(function(sender) {
        sender.track.stop();
        dc.send("disconnecting");
    });

    if (dc) {
        dc.close();
    }

    // close transceivers
    if (pc.getTransceivers) {
        pc.getTransceivers().forEach(function(transceiver) {
            if (transceiver.stop) {
                transceiver.stop();
            }
        });
    }

    // close peer connection
    setTimeout(function() {
        pc.close();
    }, 500);
    console.log('Disconnected')
}

function sdpFilterCodec(kind, codec, realSdp) {
    var allowed = []
    var rtxRegex = new RegExp('a=fmtp:(\\d+) apt=(\\d+)\r$');
    var codecRegex = new RegExp('a=rtpmap:([0-9]+) ' + escapeRegExp(codec))
    var videoRegex = new RegExp('(m=' + kind + ' .*?)( ([0-9]+))*\\s*$')
    
    var lines = realSdp.split('\n');

    var isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var match = lines[i].match(codecRegex);
            if (match) {
                allowed.push(parseInt(match[1]));
            }

            match = lines[i].match(rtxRegex);

            if (match && allowed.includes(parseInt(match[2]))) {
                allowed.push(parseInt(match[1]));
            }
        }
    }

    var skipRegex = 'a=(fmtp|rtcp-fb|rtpmap):([0-9]+)';
    var sdp = '';

    isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var skipMatch = lines[i].match(skipRegex);
            if (skipMatch && !allowed.includes(parseInt(skipMatch[2]))) {
                continue;
            } else if (lines[i].match(videoRegex)) {
                sdp += lines[i].replace(videoRegex, '$1 ' + allowed.join(' ')) + '\n';
            } else {
                sdp += lines[i] + '\n';
            }
        } else {
            sdp += lines[i] + '\n';
        }
    }

    console.log(`Processed SDP is ${sdp}`)
    sdp = sdp.replace('minptime=10;useinbandfec=1', 'minptime=10;useinbandfec=1;sprop-maxcapturerate=16000;stereo=0')
    console.log(`16kHz SDP is ${sdp}`)
    return sdp;
}

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}
