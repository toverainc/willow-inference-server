var messagesLog = document.getElementById('messages')

var wsPort = location.port;

if (window.location.protocol == 'http:') {
    var wsProto = "ws"
} else {
    var wsProto = "wss"
}

var wsURL = `${wsProto}://${window.location.hostname}:${wsPort}/api/ws/chatbot`

var ws = new WebSocket(wsURL);
console.log(`Using websocket url ${wsURL}`)

ws.onmessage = function(event) {
    messagesLog.textContent += 'Chatbot says: \n'
    messagesLog.textContent += event.data + '\n' + '\n'
};

function sendMessage(event) {
    var input = document.getElementById("messageText")
    messagesLog.textContent = 'Asking chatbot: ' + input.value + '\n' + 'Please wait...' + '\n' + '\n'
    ws.send(input.value)
    event.preventDefault()
}