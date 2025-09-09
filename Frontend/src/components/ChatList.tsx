

export default function ChatList() {
  const chats = [
    { id: 1, name: "Alex Chen", lastMessage: "Hey, how are you?", time: "2m" },
    { id: 2, name: "Design Team", lastMessage: "The new mockups look great!", time: "5m" },
    { id: 3, name: "Sarah Johnson", lastMessage: "Thanks for the feedback", time: "1h" },
    { id: 4, name: "Project Alpha", lastMessage: "Meeting scheduled for tomorrow", time: "2h" },
  ];

  return (
    <div className="chat-list flex flex-col w-[300px] bg-gray-50 border-r border-gray-300">
      <input type="text" placeholder="Search" className="search" />
      <div className="chats">
        {chats.map(chat => (
          <div key={chat.id} className="chat-item">
            <div className="avatar">{chat.name.charAt(0)}</div>
            <div className="chat-info">
              <div className="chat-name">{chat.name}</div>
              <div className="chat-last">{chat.lastMessage}</div>
            </div>
            <div className="chat-time">{chat.time}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
