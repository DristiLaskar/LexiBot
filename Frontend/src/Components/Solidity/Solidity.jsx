import { useState, useEffect } from "react";
import { Shield, Code, AlertTriangle, CheckCircle } from "lucide-react";

function Solidity() {
  const [code, setCode] = useState("");
  const [shouldFetch, setShouldFetch] = useState(false);
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState({});

  const handleSubmit = () => {
    setShouldFetch(true);
  };

  useEffect(() => {
    if (!shouldFetch) return;

    setLoading(true);

    fetch("http://127.0.0.1:8000/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ code }),
    })
      .then((res) => res.json())
      .then((data) => setOutput(data))
      .catch((err) => console.error("Error:", err))
      .finally(() => {
        setShouldFetch(false);
        setLoading(false);
      });
  }, [shouldFetch, code]);

  return (
    <div className="min-h-screen bg-slate-900 flex">
      {/* Left Panel - About */}
      <div className="w-1/3 bg-gradient-to-br from-slate-800 to-slate-900 p-8 border-r border-slate-700">
        <div className="sticky top-8">
          <div className="flex items-center gap-3 mb-8">
            <Shield className="w-8 h-8 text-blue-400" />
            <h1 className="text-3xl font-bold text-white">
              Solidity Analyzer
            </h1>
          </div>

          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <Code className="w-5 h-5 text-blue-400" />
                What We Analyze
              </h2>
              <ul className="space-y-3 text-slate-300">
                <li className="flex items-start gap-2">
                  <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                  <span>Security vulnerabilities and risk assessment</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                  <span>Smart contract best practices compliance</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                  <span>Key operations and function analysis</span>
                </li>
              </ul>
            </div>

            <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-xl p-6 border border-blue-500/20">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-yellow-400" />
                How It Works
              </h2>
              <p className="text-slate-300 leading-relaxed">
                Our AI-powered analyzer uses advanced language models to examine your Solidity code. 
                Simply paste your smart contract code in the right panel and click analyze to get 
                comprehensive insights about security, efficiency, and best practices.
              </p>
            </div>

            <div className="bg-slate-800/30 rounded-xl p-6 border border-slate-600/50">
              <h3 className="text-lg font-semibold text-white mb-3">Quick Tips</h3>
              <ul className="text-sm text-slate-400 space-y-2">
                <li>• Paste complete contract code for best results</li>
                <li>• Include all imports and dependencies</li>
                <li>• Review all security recommendations carefully</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Right Panel - Analysis */}
      <div className="flex-1 bg-slate-50 p-8">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl font-bold text-slate-800 mb-6">Code Analysis</h2>

          <div className="space-y-6">
            <div className="relative">
              <textarea
                placeholder="Paste your Solidity contract code here..."
                className="w-full bg-white rounded-xl p-6 min-h-[300px] text-sm font-mono border-2 border-slate-200 focus:border-blue-500 focus:outline-none transition-colors resize-none shadow-sm"
                value={code}
                onChange={(e) => {
                  setCode(e.target.value);
                  e.target.style.height = "auto";
                  e.target.style.height = `${e.target.scrollHeight}px`;
                }}
              />
            </div>

            <div className="flex justify-center">
              <button
                onClick={handleSubmit}
                disabled={loading || !code.trim()}
                className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-3 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
              >
                {loading ? (
                  <span className="flex items-center gap-2">
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    Analyzing...
                  </span>
                ) : (
                  "Analyze Contract"
                )}
              </button>
            </div>
          </div>

          {/* Results Section */}
          {(loading || Object.keys(output).length > 0) && (
            <div className="mt-8 space-y-6">
              <h3 className="text-xl font-semibold text-slate-800">Analysis Results</h3>
              
              {loading && (
                <div className="bg-white rounded-xl p-8 shadow-sm border border-slate-200">
                  <div className="flex items-center justify-center gap-3 text-slate-600">
                    <div className="w-6 h-6 border-2 border-slate-300 border-t-blue-500 rounded-full animate-spin"></div>
                    <span className="text-lg">Analyzing your contract...</span>
                  </div>
                </div>
              )}

              {!loading && Object.keys(output).length > 0 && (
                <div className="space-y-6">
                  {Object.entries(output).map(([key, value]) => {
                    const formatValue = (val) => {
                      if (typeof val === 'string') {
                        return val;
                      } else if (Array.isArray(val)) {
                        return val;
                      } else if (typeof val === 'object' && val !== null) {
                        return val;
                      }
                      return String(val);
                    };

                    const renderContent = (content) => {
                      if (typeof content === 'string') {
                        return (
                          <div className="prose prose-slate max-w-none">
                            <p className="text-slate-700 leading-relaxed whitespace-pre-wrap">{content}</p>
                          </div>
                        );
                      } else if (Array.isArray(content)) {
                        return (
                          <div className="space-y-2">
                            {content.map((item, idx) => (
                              <div key={idx} className="flex items-start gap-3 p-3 bg-slate-50 rounded-lg border-l-4 border-blue-400">
                                <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                                  <span className="text-xs font-semibold text-blue-600">{idx + 1}</span>
                                </div>
                                <span className="text-slate-700">{typeof item === 'object' ? JSON.stringify(item, null, 2) : item}</span>
                              </div>
                            ))}
                          </div>
                        );
                      } else if (typeof content === 'object' && content !== null) {
                        return (
                          <div className="space-y-3">
                            {Object.entries(content).map(([subKey, subValue]) => (
                              <div key={subKey} className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                                <h5 className="font-semibold text-slate-800 mb-2 capitalize">{subKey.replace('_', ' ')}</h5>
                                <div className="text-slate-700">
                                  {typeof subValue === 'string' ? (
                                    <p className="leading-relaxed">{subValue}</p>
                                  ) : (
                                    <pre className="text-sm font-mono bg-white p-3 rounded border overflow-auto">
                                      {JSON.stringify(subValue, null, 2)}
                                    </pre>
                                  )}
                                </div>
                              </div>
                            ))}
                          </div>
                        );
                      }
                      return (
                        <div className="bg-slate-50 rounded-lg p-4 font-mono text-sm text-slate-700 overflow-auto">
                          <pre className="whitespace-pre-wrap">{JSON.stringify(content, null, 2)}</pre>
                        </div>
                      );
                    };

                    const getRiskLevelColor = (risk) => {
                      if (typeof risk === 'string') {
                        const level = risk.toLowerCase();
                        if (level.includes('high') || level.includes('critical')) return 'text-red-600 bg-red-100';
                        if (level.includes('medium') || level.includes('moderate')) return 'text-yellow-600 bg-yellow-100';
                        if (level.includes('low') || level.includes('minimal')) return 'text-green-600 bg-green-100';
                      }
                      return 'text-slate-600 bg-slate-100';
                    };

                    return (
                      <div key={key} className="bg-white rounded-xl shadow-lg border border-slate-200 overflow-hidden">
                        <div className="bg-gradient-to-r from-slate-50 to-slate-100 px-6 py-4 border-b border-slate-200">
                          <h4 className="text-xl font-bold text-slate-800 capitalize flex items-center gap-3">
                            {key === 'risk_level' && <AlertTriangle className="w-6 h-6 text-yellow-500" />}
                            {key === 'security_summary' && <Shield className="w-6 h-6 text-blue-500" />}
                            {key === 'key_terms' && <Code className="w-6 h-6 text-green-500" />}
                            {key.replace('_', ' ')}
                            {key === 'risk_level' && typeof value === 'string' && (
                              <span className={`px-3 py-1 rounded-full text-sm font-semibold ${getRiskLevelColor(value)}`}>
                                {value}
                              </span>
                            )}
                          </h4>
                        </div>
                        <div className="p-6">
                          {renderContent(formatValue(value))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Solidity;