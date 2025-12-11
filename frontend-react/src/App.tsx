import { useState, useEffect, useRef } from 'react';
import { 
  Play, 
  Square, 
  Terminal, 
  Activity, 
  Wifi, 
  WifiOff,
  ShieldCheck,
  Link as LinkIcon,
  UploadCloud,
  Globe,
  Server
} from 'lucide-react';


// import chat from './api/chatApi.js'
// --- Types ---
interface LogEntry {
  id: string;
  timestamp: string;
  message: string;
  type: 'info' | 'error' | 'success';
}

// --- Main App ---
export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [urlInput, setUrlInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);


  const generateHexUUID = () => {
  const hexChars = '0123456789abcdef';
  let result = '';
  for (let i = 0; i < 32; i++) {

    result += hexChars[Math.floor(Math.random() * 16)];

  }

  return result;

};

  const session_id = generateHexUUID();

  const ws = useRef<WebSocket | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // WebSocket Connection Logic
  useEffect(() => {
    const connect = () => {
      const socket = new WebSocket(`ws://localhost:8000/ws/voice/${session_id}`);

      socket.onopen = () => {
        setIsConnected(true);
        addLog('Connected to VoiceBot Backend Bridge', 'success');
      };

      socket.onclose = () => {
        setIsConnected(false);
        setIsRunning(false);
        addLog('Disconnected from backend. Retrying...', 'error');
        setTimeout(connect, 3000);
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'log') {
            addLog(data.message, 'info');
          }
          else if (data.type === 'bot_state') {
            setIsRunning(data.running);
          }
        } catch (e) {
          console.error("Failed to parse WS message", e);
        }
      };

      ws.current = socket;
    };

    connect();

    return () => {
      ws.current?.close();
    };
  }, []);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const addLog = (message: string, type: 'info' | 'error' | 'success') => {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    setLogs(prev => [...prev, {
      id: Math.random().toString(36).substr(2, 9),
      timestamp: timeString,
      message,
      type
    }]);
  };

  const sendCommand = (command: string, payload?: any) => {
    if (ws.current && isConnected) {
      ws.current.send(JSON.stringify({ command, ...payload }));
    } else {
      addLog('WebSocket not connected', 'error');
    }
  };

  const handleStart = () => {
    sendCommand('audio');
    addLog('Sending start command...', 'info');
  };

  const handleStop = () => {
    sendCommand('stop');
    addLog('Sending stop command...', 'info');
  };

  const handleUrlUpload = () => {
    if (!urlInput.trim()) return;
    if (!urlInput.startsWith('http')) {
      addLog('Invalid URL. Please enter a valid http/https link.', 'error');
      return;
    }

    setIsUploading(true);
    addLog(`Ingesting data from: ${urlInput}`, 'info');
    sendCommand('ingest_url', { url: urlInput });
    
    setTimeout(() => {
      setIsUploading(false);
      setUrlInput('');
      addLog('URL queued for processing.', 'success');
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-[#0B0F19] bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-indigo-900/20 via-[#0B0F19] to-[#0B0F19] text-slate-200 p-4 md:p-8 flex flex-col gap-6 font-sans">
      
      {/* Header */}
      <header className="flex flex-col md:flex-row justify-between items-center bg-slate-900/40 backdrop-blur-xl border border-white/5 rounded-2xl p-6 shadow-2xl">
        <div className="flex items-center gap-4 mb-4 md:mb-0">
          <div className="bg-gradient-to-br from-blue-600 to-indigo-700 p-3 rounded-xl shadow-lg shadow-blue-500/20 border border-white/10">
            <ShieldCheck size={32} className="text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent tracking-tight">
              DCB Bank VoiceBot
            </h1>
            <div className="flex items-center gap-2 text-slate-400 text-xs font-medium uppercase tracking-wider">
              <Activity size={12} className="text-indigo-400" />
              <span>Agentic Control Interface</span>
            </div>
          </div>
        </div>

        <div className={`flex items-center gap-3 px-4 py-2 rounded-full border backdrop-blur-md transition-all duration-300 ${
          isConnected 
            ? 'bg-emerald-500/10 border-emerald-500/20 shadow-[0_0_15px_rgba(16,185,129,0.1)]' 
            : 'bg-red-500/10 border-red-500/20'
        }`}>
          {isConnected ? <Wifi size={18} className="text-emerald-400" /> : <WifiOff size={18} className="text-red-400" />}
          <span className={`text-sm font-bold ${isConnected ? 'text-emerald-400' : 'text-red-400'}`}>
            {isConnected ? 'SYSTEM ONLINE' : 'DISCONNECTED'}
          </span>
        </div>
      </header>

      {/* Main Layout */}
      <main className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1 min-h-0">
        
        {/* Left Column: Controls & Input */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          
          {/* System Controls Section */}
          <div className="bg-slate-900/40 backdrop-blur-md border border-white/5 rounded-2xl p-6 shadow-xl relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
              <Server size={100} className="text-indigo-500" />
            </div>
            
            <h3 className="text-slate-300 text-sm font-bold uppercase tracking-widest mb-6 flex items-center gap-2 relative z-10">
              <span className="w-1 h-4 bg-indigo-500 rounded-full"></span>
              System Controls
            </h3>
            
            <div className="flex gap-4 relative z-10">
              <button 
                onClick={handleStart}
                disabled={isRunning || !isConnected}
                className={`flex-1 group relative overflow-hidden flex flex-col items-center justify-center gap-3 py-8 rounded-2xl font-bold text-lg transition-all duration-300 ${
                  isRunning 
                    ? 'bg-slate-800/50 text-slate-600 cursor-not-allowed border border-slate-700/50' 
                    : 'bg-gradient-to-br from-emerald-600 to-emerald-700 text-white shadow-lg shadow-emerald-900/20 hover:shadow-emerald-500/20 hover:scale-[1.02] border border-emerald-500/20'
                }`}
              >
                <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20"></div>
                <Play size={32} fill="currentColor" className={isRunning ? 'opacity-50' : ''} />
                <span className="tracking-widest">START</span>
              </button>
              
              <button 
                onClick={handleStop}
                disabled={!isRunning || !isConnected}
                className={`flex-1 group relative overflow-hidden flex flex-col items-center justify-center gap-3 py-8 rounded-2xl font-bold text-lg transition-all duration-300 ${
                  !isRunning 
                    ? 'bg-slate-800/50 text-slate-600 cursor-not-allowed border border-slate-700/50' 
                    : 'bg-gradient-to-br from-red-600 to-rose-700 text-white shadow-lg shadow-red-900/20 hover:shadow-red-500/20 hover:scale-[1.02] border border-red-500/20'
                }`}
              >
                <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20"></div>
                <Square size={32} fill="currentColor" className={!isRunning ? 'opacity-50' : ''} />
                <span className="tracking-widest">STOP</span>
              </button>
            </div>
            
            <div className="mt-6 flex items-center justify-between text-xs font-medium text-slate-500 bg-slate-950/50 rounded-lg p-3 border border-white/5">
              <span>Status:</span>
              <span className={isRunning ? 'text-emerald-400 font-bold' : 'text-slate-400'}>
                {isRunning ? 'ACTIVE RUNNING' : 'SYSTEM IDLE'}
              </span>
            </div>
          </div>

          {/* URL Ingestion Section */}
          <div className="bg-slate-900/40 backdrop-blur-md border border-white/5 rounded-2xl p-6 shadow-xl flex-1">
            <h3 className="text-slate-300 text-sm font-bold uppercase tracking-widest mb-6 flex items-center gap-2">
              <span className="w-1 h-4 bg-cyan-500 rounded-full"></span>
              Knowledge Base
            </h3>

            <div className="space-y-4">
              <div className="relative group">
                <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-cyan-400 transition-colors">
                  <LinkIcon size={18} />
                </div>
                <input 
                  type="text" 
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  placeholder="Paste documentation URL here..."
                  disabled={!isConnected}
                  className="w-full bg-slate-950/50 border border-slate-700/50 rounded-xl py-4 pl-12 pr-4 text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50 transition-all shadow-inner"
                />
              </div>

              <button 
                onClick={handleUrlUpload}
                disabled={!isConnected || isUploading || !urlInput}
                className={`w-full py-4 rounded-xl font-bold text-sm tracking-wider uppercase transition-all flex items-center justify-center gap-2 ${
                  isUploading || !urlInput
                    ? 'bg-slate-800 text-slate-500 border border-slate-700 cursor-not-allowed'
                    : 'bg-slate-800 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500 hover:text-white hover:shadow-[0_0_20px_rgba(6,182,212,0.3)]'
                }`}
              >
                {isUploading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
                    Processing...
                  </>
                ) : (
                  <>
                    <UploadCloud size={18} />
                    Ingest Data
                  </>
                )}
              </button>

              <div className="bg-blue-900/10 border border-blue-500/10 rounded-lg p-4 mt-4">
                <div className="flex items-start gap-3">
                  <Globe size={16} className="text-blue-400 mt-0.5" />
                  <div>
                    <h4 className="text-xs font-bold text-blue-300 mb-1">RAG Context Update</h4>
                    <p className="text-[11px] text-blue-200/60 leading-relaxed">
                      URLs uploaded here are immediately processed by the Agentic RAG pipeline. The bot will reference this new data in real-time conversations.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Console */}
        <div className="lg:col-span-7 flex flex-col min-h-[500px] bg-[#090b10] rounded-2xl border border-white/10 shadow-2xl overflow-hidden relative">
          
          {/* Console Header */}
          <div className="flex items-center justify-between px-5 py-4 bg-[#0f111a] border-b border-white/5">
            <div className="flex items-center gap-3">
              <Terminal size={16} className="text-slate-400" />
              <span className="text-sm font-mono font-bold text-slate-300 tracking-tight">SYSTEM TERMINAL</span>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`}></span>
                <span className="text-[10px] font-mono text-slate-500 uppercase">{isConnected ? 'Live Stream' : 'Offline'}</span>
              </div>
              <button 
                onClick={() => setLogs([])}
                className="text-[10px] text-slate-500 hover:text-white border border-slate-800 hover:border-slate-600 px-2 py-1 rounded transition-all"
              >
                CLEAR
              </button>
            </div>
          </div>
          
          {/* Console Body */}
          <div 
            className="flex-1 p-6 font-mono text-sm overflow-y-auto custom-scrollbar relative bg-[#090b10]"
          >
            {/* Subtle Grid Background */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none"></div>

            {logs.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-slate-700/50">
                <Terminal size={48} className="mb-4 opacity-20" />
                <p className="text-sm font-medium tracking-wider">AWAITING SYSTEM OUTPUT...</p>
              </div>
            )}
            
            <div className="relative z-10 space-y-3">
              {logs.map((log) => (
                <div key={log.id} className="flex gap-4 group hover:bg-white/5 -mx-4 px-4 py-1 transition-colors">
                  <span className="text-slate-600 shrink-0 select-none text-xs pt-1 w-20 font-light">{log.timestamp}</span>
                  <div className="flex-1 break-words leading-relaxed">
                    {log.type === 'info' && <span className="text-blue-500 font-bold text-xs mr-2">INFO</span>}
                    {log.type === 'error' && <span className="text-red-500 font-bold text-xs mr-2">ERR</span>}
                    {log.type === 'success' && <span className="text-emerald-500 font-bold text-xs mr-2">OK</span>}
                    
                    <span className={`${
                      log.type === 'error' ? 'text-red-200' : 
                      log.type === 'success' ? 'text-emerald-100' : 
                      'text-slate-300'
                    }`}>
                      {log.message}
                    </span>
                  </div>
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </div>
        </div>
      </main>

      {/* Global Styles for Custom Scrollbar */}
      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #090b10; 
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #1e293b; 
          border-radius: 4px;
          border: 2px solid #090b10;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #334155; 
        }
      `}</style>
    </div>
  );
}
