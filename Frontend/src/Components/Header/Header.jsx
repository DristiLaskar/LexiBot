import { NavLink } from "react-router-dom";

function Header() {
  return (
    <div className="bg-gray-900 text-white shadow-md top-0 z-50">
      <div className="absolute left-2 text-xl font-bold py-2 px-4">
          <NavLink to="/" className="hover:text-blue-400 transition text-3xl">
            <span className="text-4xl">L</span><span className="text-4xl">eg</span><span className="text-blue-500 text-4xl">ai</span><span className="text-4xl">cy</span>
          </NavLink>
      </div>
      <nav className="relative container mx-auto h-[64px] flex items-center px-4">

        {/* Center Navigation Links */}
        <div className="absolute left-1/2 transform -translate-x-1/2 flex gap-4">
          <NavLink
            to="/"
            className={({ isActive }) =>
              isActive
                ? "bg-white text-blue-600 px-3 py-1 rounded-full font-semibold"
                : "hover:bg-blue-500 px-3 py-1 rounded-full transition"
            }
          >
            Home
          </NavLink>
          <NavLink
            to="/solidity"
            className={({ isActive }) =>
              isActive
                ? "bg-white text-blue-600 px-3 py-1 rounded-full font-semibold"
                : "hover:bg-blue-500 px-3 py-1 rounded-full transition"
            }
          >
            Solidity
          </NavLink>
          <NavLink
            to="/chatbot"
            className={({ isActive }) =>
              isActive
                ? "bg-white text-blue-600 px-3 py-1 rounded-full font-semibold"
                : "hover:bg-blue-500 px-3 py-1 rounded-full transition"
            }
          >
            LegalBOT
          </NavLink>
        </div>
      </nav>
    </div>
  );
}

export default Header;
