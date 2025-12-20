import React, { useEffect, useState, useRef } from "react";
import { BookMarked, Lightbulb, ScrollText, Earth, Bot, User } from "lucide-react";

function ChatBot() {
  const [country, setCountry] = useState("");
  const [query, setQuery] = useState("");
  const [shouldFetch, setShouldFetch] = useState(false);
  const [loading, setLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const bottomRef = useRef(null);

  useEffect(() => {
    if (!shouldFetch) return;

    setLoading(true);

    fetch("http://127.0.0.1:8000/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ country, query }),
    })
      .then((res) => res.json())
      .then((data) => {
        setChatHistory((prev) => [...prev, { user: query, bot: data.response }]);
        setQuery("");
      })
      .catch((err) => console.error("Error:", err))
      .finally(() => {
        setShouldFetch(false);
        setLoading(false);
      });
  }, [shouldFetch, country, query]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (country && query.trim() !== "") {
      setShouldFetch(true);
    } 
  };

  const handleExampleClick = (exampleQuery) => {
    setQuery(exampleQuery);
  };

  const formatBotResponse = (text) => {
    if (!text) return "";

    const paragraphs = text.split('\n\n').filter(p => p.trim());

    const renderWithBold = (content) => {
      const parts = content.split(/(\*+[^*]+\*+)/g);
      return parts.map((part, i) => {
        const match = part.match(/^\*+(.+?)\*+$/);
        if (match) {
          return (
            <strong key={i} className="font-semibold text-gray-900">
              {match[1]}
            </strong>
          );
        }
        return <span key={i}>{part}</span>;
      });
    };

    return paragraphs
      .filter(p => !/^Disclaimer:/i.test(p.trim()))
      .map((paragraph, index) => {
        if (paragraph.match(/^\d+\.|^[-•*]/m)) {
          const listItems = paragraph.split('\n').filter(item => item.trim());
          return (
            <div key={index} className="mb-4">
              <ul className="space-y-2">
                {listItems.map((item, idx) => (
                  <li key={idx} className="flex items-start space-x-2">
                    <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></span>
                    <span className="text-gray-700 leading-relaxed">
                      {renderWithBold(item.replace(/^\d+\.|^[-•*]\s*/, ''))}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          );
        }

        if (paragraph.match(/^(Important|Note|Warning|Summary):/i)) {
          return (
            <div key={index} className="mb-4 p-3 bg-amber-50 border-l-4 border-amber-400 rounded-r-lg">
              <p className="text-amber-800 font-medium leading-relaxed">
                {renderWithBold(paragraph)}
              </p>
            </div>
          );
        }

        return (
          <p key={index} className="mb-4 text-gray-700 leading-relaxed">
            {renderWithBold(paragraph)}
          </p>
        );
      });
  };  

  return (
    <div className="min-h-screen bg-slate-900 flex">
      {/* Sidebar */}
      <div className="w-1/3 bg-gray-800 p-6 shadow-lg space-y-6">
        <div className="flex space-x-3">
          <BookMarked className="w-8 h-8 text-yellow-400" />
          <h1 className="text-2xl font-bold text-white">LegalBOT</h1>
        </div>

        <div className="bg-gray-700/80 p-4 rounded-2xl text-gray-300">
          LegalBOT is an AI-powered legal assistant that provides information
          about legal matters in different countries. It is for informational purposes only.
        </div>

        <div className="flex space-x-3 mt-9">
          <Earth className="w-7 h-10 text-green-600" />
          <select
            className="p-2 border rounded bg-white"
            value={country}
            onChange={(e) => setCountry(e.target.value)}
          >
            <option value="">Select a country</option>
            <option value="USA">USA</option>
            <option value="India">India</option>
            <option value="Germany">Germany</option>
          </select>
        </div>

        <div className="flex space-x-3 mt-10">
          <Lightbulb className="w-8 h-8 text-yellow-400" />
          <h1 className="text-2xl font-bold text-white">Example Queries</h1>
        </div>   

        {[
          "What are the Miranda Rights?",
          "What are intellectual property rights?",
          "What are the essential elements of a valid contract?",
        ].map((q, i) => (
          <div
            key={i}
            className="bg-gray-700/50 flex space-x-3 p-2 rounded cursor-pointer hover:bg-gray-700/70 transition-colors"
            onClick={() => handleExampleClick(q)}
          >
            <ScrollText className="w-5 h-5 text-blue-600" />
            <h4 className="text-white">{q}</h4>
          </div>
        ))}
      </div>

      {/* Chat area */}
      <div className="w-2/3 bg-white flex flex-col p-5">
        <div className="flex-grow overflow-y-auto pr-2 flex flex-col">
          {chatHistory.length === 0 ? (
            <div className="flex flex-grow justify-center items-center">
              <div className="text-center">
                <Bot className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h1 className="font-bold text-gray-500 text-4xl mb-2">We are here to help</h1>
                <p className="text-gray-400 text-lg">Ask me anything about legal matters</p>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {chatHistory.map((item, index) => (
                <div key={index} className="space-y-4">
                  <div className="flex justify-end">
                    <div className="bg-gradient-to-r from-blue-500 to-blue-600 p-4 rounded-2xl max-w-[80%] shadow-lg">
                      <div className="flex items-start space-x-2">
                        <User className="w-5 h-5 text-white mt-0.5 flex-shrink-0" />
                        <p className="text-white font-medium whitespace-pre-wrap leading-relaxed">{item.user}</p>
                      </div>
                    </div>
                  </div>

                  <div className="flex justify-start">
                    <div className="bg-gradient-to-r from-gray-50 to-gray-100 p-5 rounded-2xl max-w-[85%] shadow-lg border border-gray-200">
                      <div className="flex items-start space-x-3">
                        <div className="bg-blue-100 p-2 rounded-full flex-shrink-0">
                          <Bot className="w-5 h-5 text-blue-600" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="prose prose-sm max-w-none">
                            {formatBotResponse(item.bot)}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}

              {loading && (
                <div className="flex justify-start">
                  <div className="bg-gradient-to-r from-gray-50 to-gray-100 p-5 rounded-2xl shadow-lg border border-gray-200">
                    <div className="flex items-center space-x-3">
                      <div className="bg-blue-100 p-2 rounded-full">
                        <Bot className="w-5 h-5 text-blue-600" />
                      </div>
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div ref={bottomRef}></div>
            </div>
          )}
        </div>

        <div className="mt-4">
          <textarea
            placeholder="Write your legal query here..."
            className="w-full bg-white rounded-xl p-4 text-sm font-mono border-2 border-slate-200 focus:border-blue-500 focus:outline-none transition-colors resize-none shadow-sm min-h-[60px]"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              e.target.style.height = "auto";
              e.target.style.height = `${e.target.scrollHeight}px`;
            }}
            disabled={loading}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
          />

          <div className="flex space-x-2 mt-2 justify-end">
            <button
              onClick={handleSubmit}
              className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-xl hover:from-blue-700 hover:to-blue-800 transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none shadow-lg"
              disabled={loading || !country || !query.trim()}
            >
              <ScrollText className="w-5 h-5" />
              <span>{loading ? "Sending..." : "Send Query"}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ChatBot;
