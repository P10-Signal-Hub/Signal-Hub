import Topbar from "./components/Topbar";
import Sidebar from "./components/Sidebar";
import ChatList from "./components/ChatList";
import ChatWindow from "./components/ChatWindow";
import "./styles/theme.css";

export default function App() {
  return (
    <div className="app">
      <Topbar />
      <Sidebar />
      <div className="chat-area">
        <ChatList />
        <ChatWindow />
      </div>
    </div>
  );
}