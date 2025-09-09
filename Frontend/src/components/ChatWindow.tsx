
export default function ChatWindow() {
  const messages = [
    { id: 1, sender: "Alex Chen", text: "Hey there! How are you doing today?", time: "10:30 AM", fromMe: false },
    { id: 2, sender: "Me", text: "I’m doing great, thanks for asking! Just working on some new designs.", time: "10:32 AM", fromMe: true },
    { id: 3, sender: "Alex Chen", text: "That sounds interesting! What kind of designs are you working on?", time: "10:33 AM", fromMe: false },
    { id: 4, sender: "Me", text: "I’m working on a new chat interface design. It’s been quite challenging but fun!", time: "10:35 AM", fromMe: true },
  ];

  return (
    
    <div className="chat-window flex-1 flex flex-col bg-white border-l border-gray-200">

      <div className="chat-header flex justify-between p-2 border-b border-gray-300">
        <div className="chat-user">Alex Chen</div>
        <div className="chat-status">Online</div>
      </div>

      <div className="messages">
        {messages.map(msg => (
          <div key={msg.id} className={`message ${msg.fromMe ? "me" : "other"}`}>
            {!msg.fromMe && <div className="username">{msg.sender}</div>}
            <div className="bubble">{msg.text}</div>
            <div className="time">{msg.time}</div>
          </div>
        ))}
      </div>

      <div className="chat-input">
        <input type="text" placeholder="Message..." />
        <button>Send</button>
      </div>
    </div>
  );
}
