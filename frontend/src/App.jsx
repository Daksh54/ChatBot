import React, { useState, useEffect, useRef } from "react";
import axios from "axios";

const API_BASE_URL = "http://localhost:8000";

const App = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [fileHash, setFileHash] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [summaryQuery, setSummaryQuery] = useState("");
  const [theme, setTheme] = useState("Dark");

  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // ---------------- THEME ----------------
  const themes = {
    Dark: {
      bg: "#121212",
      chatBg: "#1f1f1f",
      userBg: "#4caf50",
      userText: "#fff",
      botBg: "#333",
      botText: "#f1f1f1",
      btn: "#4caf50",
      btnText: "#fff",
      text: "#fff",
      border: "#444",
    },
    Light: {
      bg: "#f8f8f8",
      chatBg: "#fff",
      userBg: "#4caf50",
      userText: "#fff",
      botBg: "#e0e0e0",
      botText: "#000",
      btn: "#4caf50",
      btnText: "#fff",
      text: "#000",
      border: "#ccc",
    },
    Midnight: {
      bg: "#0b0c10",
      chatBg: "#1f2833",
      userBg: "#45a29e",
      userText: "#0b0c10",
      botBg: "#0b0c10",
      botText: "#c5c6c7",
      btn: "#66fcf1",
      btnText: "#0b0c10",
      text: "#c5c6c7",
      border: "#1f2833",
    },
  };
  const colors = themes[theme];

  // ---------------- EFFECTS ----------------
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  useEffect(() => {
    if (fileHash) {
      fetchChatHistory(fileHash);
    }
  }, [fileHash]);

  // ---------------- BACKEND INTEGRATION ----------------
  const fetchChatHistory = async (hash) => {
    try {
      const res = await axios.get(`${API_BASE_URL}/history?file_hash=${hash}`);
      setMessages(
        res.data.history.map((m) => [
          { type: "user", content: m.user },
          { type: "bot", content: m.bot },
        ]).flat()
      );
    } catch {
      setMessages([]);
    }
  };

  const uploadFile = async (event) => {
    const selected = event.target.files[0];
    if (!selected) return;
    setUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", selected);
      const res = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      if (res.status === 200 && res.data.file_hash) {
        setFile(selected);
        setFileName(selected.name);
        setFileHash(res.data.file_hash);
        setMessages([]);
      } else throw new Error(res.data.error || "Upload failed");
    } catch (e) {
      alert("Upload error: " + e.message);
    } finally {
      setUploading(false);
    }
  };

  const sendMessage = async () => {
    if (!input.trim()) return;
    const text = input.trim();
    setMessages((prev) => [...prev, { type: "user", content: text }]);
    setInput("");
    setLoading(true);

    try {
      const res = await axios.post(`${API_BASE_URL}/chat`, {
        message: text,
        file_hash: fileHash,
      });
      setMessages((prev) => [
        ...prev,
        { type: "bot", content: res.data.reply || res.data.response },
      ]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { type: "bot", content: "Error: " + e.message },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const sendSummary = async () => {
    const q = summaryQuery.trim();
    if (!fileHash) return alert("Upload a document first.");
    setLoading(true);
    setMessages((prev) => [
      ...prev,
      { type: "user", content: q || "[Full Document Summary]" },
    ]);

    try {
      const res = await axios.post(`${API_BASE_URL}/chat`, {
        file_hash: fileHash,
        message: q || "Summarize this document",
      });
      setMessages((prev) => [
        ...prev,
        { type: "bot", content: res.data.response },
      ]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { type: "bot", content: "Summary error: " + e.message },
      ]);
    } finally {
      setSummaryQuery("");
      setLoading(false);
    }
  };

  const clearChat = async () => {
    if (!fileHash) {
      setMessages([]);
      return;
    }
    try {
      await axios.post(`${API_BASE_URL}/clear`, { file_hash: fileHash });
      setMessages([]);
      setFileHash("");
      setFileName("");
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch {
      alert("Error clearing chat");
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleSummaryKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendSummary();
    }
  };

  // ---------------- UI ----------------
  return (
    <div
      style={{
        backgroundColor: colors.bg,
        color: colors.text,
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "20px",
        fontFamily: "Inter, sans-serif",
      }}
    >
      <div style={{ maxWidth: "700px", width: "100%" }}>
        {/* Header */}
        <header
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "15px",
          }}
        >
          <h2>🤖 ChatBot</h2>
          <select
            value={theme}
            onChange={(e) => setTheme(e.target.value)}
            style={{
              background: colors.chatBg,
              color: colors.text,
              border: `1px solid ${colors.border}`,
              borderRadius: "6px",
              padding: "6px",
            }}
          >
            <option value="Dark">Dark</option>
            <option value="Light">Light</option>
            <option value="Midnight">Midnight</option>
          </select>
        </header>

        {/* File upload */}
        <div style={{ marginBottom: "10px" }}>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.png,.jpg,.jpeg,.txt,.docx"
            onChange={uploadFile}
            disabled={uploading}
          />
          {uploading && <p>Processing file...</p>}
          {fileName && <p>📄 {fileName}</p>}
        </div>

        {/* Summary input */}
        {fileHash && (
          <div style={{ display: "flex", marginBottom: "10px" }}>
            <input
              type="text"
              placeholder="Ask about the file or press Enter for summary"
              value={summaryQuery}
              onChange={(e) => setSummaryQuery(e.target.value)}
              onKeyPress={handleSummaryKeyPress}
              style={{
                flex: 1,
                background: colors.chatBg,
                color: colors.text,
                border: `1px solid ${colors.border}`,
                borderRadius: "6px",
                padding: "10px",
              }}
            />
            <button
              onClick={sendSummary}
              disabled={loading}
              style={{
                marginLeft: "8px",
                background: colors.btn,
                color: colors.btnText,
                border: "none",
                borderRadius: "6px",
                padding: "10px 20px",
                cursor: "pointer",
              }}
            >
              {loading ? "..." : "Send"}
            </button>
          </div>
        )}

        {/* Chat box */}
        <div
          style={{
            backgroundColor: colors.chatBg,
            border: `1px solid ${colors.border}`,
            borderRadius: "10px",
            height: "400px",
            overflowY: "auto",
            padding: "10px",
            marginBottom: "10px",
          }}
        >
          {messages.map((m, i) => (
            <div
              key={i}
              style={{
                backgroundColor:
                  m.type === "user" ? colors.userBg : colors.botBg,
                color: m.type === "user" ? colors.userText : colors.botText,
                borderRadius: "10px",
                padding: "10px 14px",
                margin: "8px 0",
                maxWidth: "75%",
                alignSelf: m.type === "user" ? "flex-end" : "flex-start",
                float: m.type === "user" ? "right" : "left",
                clear: "both",
              }}
            >
              {m.content}
            </div>
          ))}
          {loading && (
            <div
              style={{
                background: colors.botBg,
                color: colors.botText,
                padding: "10px",
                borderRadius: "10px",
                marginBottom: "10px",
                width: "fit-content",
              }}
            >
              Thinking...
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Message input */}
        <div style={{ display: "flex" }}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type a message..."
            style={{
              flex: 1,
              background: colors.chatBg,
              color: colors.text,
              border: `1px solid ${colors.border}`,
              borderRadius: "6px",
              padding: "10px",
            }}
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            style={{
              marginLeft: "8px",
              background: colors.btn,
              color: colors.btnText,
              border: "none",
              borderRadius: "6px",
              padding: "10px 20px",
              cursor: "pointer",
            }}
          >
            Send
          </button>
        </div>

        {/* Clear chat */}
        <div style={{ marginTop: "10px", textAlign: "center" }}>
          <button
            onClick={clearChat}
            style={{
              background: "#e53935",
              color: "#fff",
              border: "none",
              borderRadius: "6px",
              padding: "8px 20px",
              cursor: "pointer",
            }}
          >
            🗑️ Clear Chat
          </button>
        </div>
      </div>
    </div>
  );
};

export default App;
