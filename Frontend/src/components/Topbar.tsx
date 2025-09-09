
export default function Topbar() {
  return (
    <header className="header">
      <div className="hub-selector">
        <select>
          <option>General Hub</option>
          <option>Design Team</option>
          <option>Project Alpha</option>
          <option>Development</option>
        </select>
      </div>
      <button className="button">+ New Hub</button>
    </header>
  );
}
