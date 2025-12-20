
import { Shield, Globe, ScrollText } from "lucide-react";
import { Link } from "react-router-dom";

function Home() {
  return (
    <div className="bg-slate-900 text-white min-h-screen flex items-center justify-center px-6 ">
      <div className="max-w-3xl text-center space-y-8">
        <h1 className="text-4xl font-extrabold leading-tight">
          Unlock the True Potential of Smart Contracts. <br />
          <span className="text-blue-400">With Confidence.</span>
        </h1>

        <p className="text-lg text-slate-300">
          Navigating the world of smart contracts can be complex — but it doesn't have to be.
          We've built a powerful platform to bring <span className="text-white font-semibold">clarity</span> and
          <span className="text-white font-semibold"> security</span> to every line of your code.
        </p>

        <div className="grid gap-6 sm:grid-cols-full text-left">
          <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-md">
            <Shield className="text-blue-400 w-6 h-6 mb-3" />
            <h3 className="font-bold text-lg mb-1">In-Depth Security Analysis</h3>
            <p className="text-slate-400">
              Detect risks, summarize complex logic, and highlight key contract features instantly.
            </p>
          </div>
          
          <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-md col-span-full">
            <Globe className="text-green-400 w-6 h-6 mb-3" />
            <h3 className="font-bold text-lg mb-1">Meet LegalBOT</h3>
            <p className="text-slate-400">
              Our smart chatbot gives legal insights based on your country — whether you’re a developer, business, or individual.
            </p>
          </div>
        </div>

        <div className="flex space-x-3 justify-center">
          <Link
            to="/solidity"
            className="inline-block mt-6 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-semibold transition"
          >
            Analyze Your Contract
          </Link>
          <Link
            to="/chatbot"
            className="inline-block mt-6 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-semibold transition"
          >
            Ask your query
          </Link>
        </div>
      </div>
    </div>
  );
}

export default Home;
