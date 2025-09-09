import { FiMessageSquare, FiBell, FiSettings, FiUser } from "react-icons/fi";

export default function Sidebar() {
  return (
    <div className="sidebar flex flex-col items-center h-full bg-[#0d1b2a] border-r border-gray-200">
      <div className="profile-icon">JD</div>
      <div className="menu-icons">
        <FiMessageSquare className="icon active" />
        <FiBell className="icon" />
        <FiUser className="icon" />
        <FiSettings className="icon" />
      </div>
    </div>
  );
}