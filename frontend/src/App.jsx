// Author: Daksh Sharma 26434

import { useEffect, useEffectEvent, useRef, useState } from "react";
import axios from "axios";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import {
  ArrowUp,
  Bot,
  BrainCircuit,
  DatabaseZap,
  FileUp,
  FolderKanban,
  LoaderCircle,
  MessageSquareText,
  PanelLeftClose,
  Plus,
  Sparkles,
} from "lucide-react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const createMessageId = () =>
  typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random()}`;

function CopyCodeButton({ code }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-200 transition hover:bg-white/10"
    >
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

function MarkdownMessage({ children }) {
  return (
    <Markdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ inline, className, children: codeChildren, ...props }) {
          const match = /language-(\w+)/.exec(className || "");
          const code = String(codeChildren).replace(/\n$/, "");

          if (inline) {
            return (
              <code className="rounded bg-white/10 px-1.5 py-0.5 text-sm" {...props}>
                {codeChildren}
              </code>
            );
          }

          return (
            <div className="my-4 overflow-hidden rounded-2xl border border-white/10">
              <div className="flex items-center justify-between bg-slate-950 px-4 py-2 text-xs uppercase tracking-[0.2em] text-slate-400">
                <span>{match?.[1] || "code"}</span>
                <CopyCodeButton code={code} />
              </div>
              <SyntaxHighlighter
                {...props}
                language={match?.[1]}
                style={oneDark}
                customStyle={{ margin: 0, borderRadius: 0, background: "#020617" }}
              >
                {code}
              </SyntaxHighlighter>
            </div>
          );
        },
        table({ children }) {
          return (
            <div className="my-4 overflow-x-auto">
              <table className="min-w-full text-sm">{children}</table>
            </div>
          );
        },
        thead({ children }) {
          return <thead className="bg-white/5">{children}</thead>;
        },
        th({ children }) {
          return <th className="border border-white/10 px-3 py-2 text-left font-medium">{children}</th>;
        },
        td({ children }) {
          return <td className="border border-white/10 px-3 py-2 align-top">{children}</td>;
        },
      }}
    >
      {children}
    </Markdown>
  );
}

export default function App() {
  const [workspaceName, setWorkspaceName] = useState("");
  const [messageInput, setMessageInput] = useState("");
  const [workspaces, setWorkspaces] = useState([]);
  const [selectedWorkspaceId, setSelectedWorkspaceId] = useState("");
  const [documents, setDocuments] = useState([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState("");
  const [messages, setMessages] = useState([]);
  const [statusText, setStatusText] = useState("Hybrid retrieval ready.");
  const [isCreatingWorkspace, setIsCreatingWorkspace] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeTask, setActiveTask] = useState(null);
  const [viewerPage, setViewerPage] = useState(1);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);
  const bootstrappedRef = useRef(false);

  const selectedWorkspace = workspaces.find((workspace) => workspace.id === selectedWorkspaceId) || null;
  const selectedDocument = documents.find((document) => document.id === selectedDocumentId) || null;

  const scrollChatToBottom = useEffectEvent(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  });

  useEffect(() => {
    scrollChatToBottom();
  }, [messages, scrollChatToBottom]);

  useEffect(() => {
    if (bootstrappedRef.current) {
      return;
    }

    bootstrappedRef.current = true;
    void bootstrap();
  }, []);

  useEffect(() => {
    if (!selectedWorkspaceId) {
      setDocuments([]);
      setMessages([]);
      return;
    }

    void (async () => {
      try {
        await Promise.all([
          loadDocuments(selectedWorkspaceId),
          loadMessages(selectedWorkspaceId),
        ]);
      } catch (error) {
        setStatusText(`Workspace load failed: ${error.message}`);
      }
    })();
  }, [selectedWorkspaceId]);

  useEffect(() => {
    if (!activeTask?.id || activeTask.status === "SUCCEEDED" || activeTask.status === "FAILED") {
      return;
    }

    const intervalId = window.setInterval(() => {
      void pollTaskStatus(activeTask.id);
    }, 2000);

    void pollTaskStatus(activeTask.id);

    return () => window.clearInterval(intervalId);
  }, [activeTask?.id, activeTask?.status, selectedWorkspaceId]);

  async function bootstrap() {
    try {
      const { data } = await axios.get(`${API_BASE_URL}/workspaces`);
      const items = data.items || [];
      setWorkspaces(items);
      if (items[0]) {
        setSelectedWorkspaceId(items[0].id);
      }
      if (!items.length) {
        const created = await createWorkspace("Flagship Workspace", "Primary research lane for documents and chats.");
        setSelectedWorkspaceId(created.id);
      }
    } catch (error) {
      setStatusText(`Unable to load workspace list: ${error.message}`);
    }
  }

  async function createWorkspace(name, description) {
    const { data } = await axios.post(`${API_BASE_URL}/workspaces`, {
      name,
      description,
    });
    setWorkspaces((prev) => {
      const exists = prev.some((workspace) => workspace.id === data.id);
      if (exists) {
        return prev.map((workspace) => (workspace.id === data.id ? data : workspace));
      }
      return [data, ...prev];
    });
    return data;
  }

  async function loadDocuments(workspaceId) {
    const { data } = await axios.get(`${API_BASE_URL}/workspaces/${workspaceId}/documents`);
    const items = data.items || [];
    setDocuments(items);
    setSelectedDocumentId((current) => {
      if (current && items.some((document) => document.id === current)) {
        return current;
      }
      return items[0]?.id || "";
    });
  }

  async function loadMessages(workspaceId) {
    const { data } = await axios.get(`${API_BASE_URL}/workspaces/${workspaceId}/messages`);
    const history = [];

    for (const item of data.items || []) {
      history.push({
        id: `${item.id}-user`,
        role: "user",
        content: item.user_message,
        citations: [],
      });
      history.push({
        id: `${item.id}-assistant`,
        role: "assistant",
        content: item.assistant_message,
        citations: item.citations || [],
      });
    }

    setMessages(history);
  }

  function upsertDocument(document) {
    if (!document) {
      return;
    }

    setDocuments((prev) => {
      const exists = prev.some((item) => item.id === document.id);
      if (exists) {
        return prev.map((item) => (item.id === document.id ? document : item));
      }
      return [document, ...prev];
    });
  }

  async function pollTaskStatus(taskId) {
    try {
      const { data } = await axios.get(`${API_BASE_URL}/tasks/${taskId}`);
      const nextTask = data.task;
      setActiveTask(nextTask);
      upsertDocument(data.document);

      if (nextTask.status === "PROCESSING") {
        setStatusText(`${nextTask.phase} (${nextTask.progress}%)`);
      }

      if (nextTask.status === "SUCCEEDED") {
        if (selectedWorkspaceId) {
          await loadDocuments(selectedWorkspaceId);
        }
        if (data.document?.id) {
          setSelectedDocumentId(data.document.id);
        }
        setViewerPage(1);
        setStatusText(`${data.document?.name || "Document"} indexed successfully.`);
      }

      if (nextTask.status === "FAILED") {
        setStatusText(`Indexing failed: ${nextTask.error || "Unknown worker error"}`);
      }
    } catch (error) {
      setStatusText(`Task polling failed: ${error.message}`);
    }
  }

  async function handleCreateWorkspace() {
    if (!workspaceName.trim() || isCreatingWorkspace) {
      return;
    }

    setIsCreatingWorkspace(true);
    try {
      const workspace = await createWorkspace(
        workspaceName.trim(),
        "A dedicated collection for portfolio-grade retrieval workflows."
      );
      setWorkspaceName("");
      setSelectedWorkspaceId(workspace.id);
      setStatusText(`Workspace "${workspace.name}" is ready.`);
    } catch (error) {
      setStatusText(`Workspace creation failed: ${error.message}`);
    } finally {
      setIsCreatingWorkspace(false);
    }
  }

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file || !selectedWorkspaceId) {
      return;
    }

    setIsUploading(true);
    setStatusText(`Queueing ${file.name} for background indexing...`);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("workspace_id", selectedWorkspaceId);
      const { data } = await axios.post(`${API_BASE_URL}/documents/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setActiveTask(data.task);
      upsertDocument(data.document);
      setSelectedDocumentId(data.document.id);
      setViewerPage(1);
      setStatusText(`${data.document.name} accepted. Worker is indexing it now.`);
    } catch (error) {
      setStatusText(`Upload failed: ${error.message}`);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  async function handleSendMessage() {
    const trimmed = messageInput.trim();
    if (!trimmed || !selectedWorkspaceId || isStreaming) {
      return;
    }

    const userMessage = {
      id: createMessageId(),
      role: "user",
      content: trimmed,
      citations: [],
    };
    const assistantMessageId = createMessageId();

    setMessages((prev) => [
      ...prev,
      userMessage,
      { id: assistantMessageId, role: "assistant", content: "", citations: [], streaming: true },
    ]);
    setMessageInput("");
    setIsStreaming(true);
    setStatusText("Generating streamed answer...");

    try {
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspace_id: selectedWorkspaceId,
          document_id: selectedDocumentId || null,
          message: trimmed,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const eventChunk of events) {
          const line = eventChunk
            .split("\n")
            .find((entry) => entry.startsWith("data: "));

          if (!line) {
            continue;
          }

          const payload = JSON.parse(line.slice(6));

          if (payload.type === "context") {
            setMessages((prev) =>
              prev.map((message) =>
                message.id === assistantMessageId
                  ? { ...message, citations: payload.citations || [] }
                  : message
              )
            );
          }

          if (payload.type === "assistant_chunk") {
            setMessages((prev) =>
              prev.map((message) =>
                message.id === assistantMessageId
                  ? { ...message, content: `${message.content}${payload.content}` }
                  : message
              )
            );
          }

          if (payload.type === "error") {
            throw new Error(payload.message);
          }

          if (payload.type === "assistant_done") {
            setMessages((prev) =>
              prev.map((message) =>
                message.id === assistantMessageId ? { ...message, streaming: false } : message
              )
            );
            setStatusText("Answer complete.");
          }
        }
      }
    } catch (error) {
      setMessages((prev) =>
        prev.map((message) =>
          message.id === assistantMessageId
            ? {
                ...message,
                content: `Streaming error: ${error.message}`,
                streaming: false,
              }
            : message
        )
      );
      setStatusText(`Streaming failed: ${error.message}`);
    } finally {
      setIsStreaming(false);
    }
  }

  function handleMessageKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSendMessage();
    }
  }

  function jumpToCitation(citation) {
    setViewerPage(citation.page_start || 1);
    if (citation.document_id) {
      setSelectedDocumentId(citation.document_id);
    }
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.16),_transparent_32%),linear-gradient(135deg,_#020617,_#0f172a_46%,_#111827)] text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-[1600px] gap-4 px-4 py-4 lg:px-6">
        <aside
          className={`relative overflow-hidden rounded-[2rem] border border-white/10 bg-slate-950/55 shadow-2xl shadow-cyan-950/30 backdrop-blur-xl transition-all ${
            sidebarCollapsed ? "w-[88px]" : "w-full max-w-sm"
          }`}
        >
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,_rgba(14,165,233,0.18),_transparent_30%),linear-gradient(180deg,_rgba(15,23,42,0.2),_rgba(2,6,23,0.85))]" />
          <div className="relative flex h-full flex-col p-5">
            <div className="mb-6 flex items-start justify-between gap-3">
              <div className={sidebarCollapsed ? "hidden" : "block"}>
                <p className="text-xs uppercase tracking-[0.35em] text-cyan-300/80">NexusRAG</p>
                <h1 className="mt-2 font-display text-3xl font-semibold leading-tight text-white">
                  Portfolio-grade
                  <br />
                  document intelligence
                </h1>
              </div>
              <button
                type="button"
                onClick={() => setSidebarCollapsed((value) => !value)}
                className="rounded-2xl border border-white/10 bg-white/5 p-3 text-slate-200 transition hover:bg-white/10"
              >
                <PanelLeftClose className="h-4 w-4" />
              </button>
            </div>

            <div className={sidebarCollapsed ? "hidden" : "block"}>
              <div className="rounded-3xl border border-cyan-400/15 bg-cyan-400/10 p-4 text-sm text-cyan-50">
                <div className="flex items-center gap-3">
                  <Sparkles className="h-5 w-5 text-cyan-300" />
                  <p className="font-medium">Hybrid retrieval + streaming answers</p>
                </div>
                <p className="mt-2 text-cyan-100/80">
                  Mongo-backed workspaces, Qdrant vectors, and source-aware citations are wired into the new shell.
                </p>
              </div>

              <div className="mt-6">
                <p className="mb-3 text-xs uppercase tracking-[0.3em] text-slate-400">Create workspace</p>
                <div className="flex gap-2">
                  <input
                    value={workspaceName}
                    onChange={(event) => setWorkspaceName(event.target.value)}
                    placeholder="Research workspace"
                    className="flex-1 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-400/50 focus:bg-white/10"
                  />
                  <button
                    type="button"
                    onClick={() => void handleCreateWorkspace()}
                    disabled={isCreatingWorkspace}
                    className="inline-flex items-center gap-2 rounded-2xl bg-cyan-400 px-4 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    <Plus className="h-4 w-4" />
                    Add
                  </button>
                </div>
              </div>

              <div className="mt-6 space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Workspaces</p>
                  <FolderKanban className="h-4 w-4 text-slate-500" />
                </div>
                <div className="space-y-2">
                  {workspaces.map((workspace) => {
                    const isActive = workspace.id === selectedWorkspaceId;
                    return (
                      <button
                        key={workspace.id}
                        type="button"
                        onClick={() => setSelectedWorkspaceId(workspace.id)}
                        className={`w-full rounded-2xl border px-4 py-3 text-left transition ${
                          isActive
                            ? "border-cyan-300/40 bg-cyan-300/12 text-white shadow-lg shadow-cyan-950/30"
                            : "border-white/10 bg-white/5 text-slate-300 hover:bg-white/10"
                        }`}
                      >
                        <p className="font-medium">{workspace.name}</p>
                        <p className="mt-1 text-xs text-slate-400">
                          {workspace.description || "Workspace for cross-document reasoning"}
                        </p>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="mt-6">
                <div className="flex items-center justify-between">
                  <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Documents</p>
                  <FileUp className="h-4 w-4 text-slate-500" />
                </div>
                <label className="mt-3 flex cursor-pointer items-center justify-center gap-2 rounded-2xl border border-dashed border-cyan-300/35 bg-white/5 px-4 py-4 text-sm text-slate-200 transition hover:bg-white/10">
                  {isUploading ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <FileUp className="h-4 w-4" />}
                  Upload PDF, DOCX, CSV, XLSX, TXT or image
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,.docx,.txt,.png,.jpg,.jpeg,.csv,.xlsx,.xls"
                    onChange={(event) => void handleUpload(event)}
                    className="hidden"
                  />
                </label>
                <div className="mt-3 space-y-2">
                  {activeTask && (
                    <div className="rounded-2xl border border-cyan-300/20 bg-cyan-400/10 px-4 py-4">
                      <div className="flex items-center justify-between gap-3">
                        <p className="text-sm font-medium text-white">
                          {activeTask.status === "SUCCEEDED" ? "Indexing complete" : "Indexing in progress"}
                        </p>
                        <p className="text-xs uppercase tracking-[0.2em] text-cyan-100/80">
                          {activeTask.progress}%
                        </p>
                      </div>
                      <p className="mt-2 text-sm text-cyan-50/80">
                        {activeTask.phase}
                      </p>
                      <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-900/70">
                        <div
                          className={`h-full rounded-full ${
                            activeTask.status === "FAILED" ? "bg-rose-400" : "bg-cyan-300"
                          }`}
                          style={{ width: `${Math.max(6, activeTask.progress || 0)}%` }}
                        />
                      </div>
                    </div>
                  )}
                  {documents.map((document) => {
                    const isSelected = document.id === selectedDocumentId;
                    return (
                      <button
                        key={document.id}
                        type="button"
                        onClick={() => {
                          setSelectedDocumentId(document.id);
                          setViewerPage(1);
                        }}
                        className={`w-full rounded-2xl border px-4 py-3 text-left transition ${
                          isSelected
                            ? "border-sky-400/50 bg-sky-400/12 text-white"
                            : "border-white/10 bg-white/5 text-slate-300 hover:bg-white/10"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-3">
                          <p className="truncate font-medium">{document.name}</p>
                          <span
                            className={`rounded-full px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] ${
                              document.status === "READY"
                                ? "bg-emerald-400/15 text-emerald-200"
                                : document.status === "FAILED"
                                  ? "bg-rose-400/15 text-rose-200"
                                  : "bg-amber-400/15 text-amber-200"
                            }`}
                          >
                            {document.status || "READY"}
                          </span>
                        </div>
                        <p className="mt-1 text-xs text-slate-400">
                          {document.page_count} pages • {document.chunk_count} chunks
                        </p>
                      </button>
                    );
                  })}
                  {!documents.length && (
                    <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-6 text-sm text-slate-400">
                      No documents in this workspace yet.
                    </div>
                  )}
                </div>
              </div>
            </div>

            {sidebarCollapsed && (
              <div className="mt-8 flex flex-1 flex-col items-center gap-4">
                <div className="rounded-2xl border border-cyan-400/20 bg-cyan-400/10 p-3">
                  <BrainCircuit className="h-5 w-5 text-cyan-300" />
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                  <FolderKanban className="h-5 w-5 text-slate-300" />
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                  <FileUp className="h-5 w-5 text-slate-300" />
                </div>
              </div>
            )}
          </div>
        </aside>

        <main className="grid flex-1 gap-4 lg:grid-cols-[1.12fr_0.88fr]">
          <section className="rounded-[2rem] border border-white/10 bg-slate-950/45 p-5 shadow-2xl shadow-slate-950/40 backdrop-blur-xl">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.35em] text-slate-400">Evidence viewer</p>
                <h2 className="mt-2 text-2xl font-semibold text-white">
                  {selectedDocument ? selectedDocument.name : "Select a document"}
                </h2>
              </div>
              <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-300">
                Page {viewerPage}
              </div>
            </div>

            <div className="mt-5 grid gap-4 xl:grid-cols-[1fr_320px]">
              <div className="min-h-[560px] overflow-hidden rounded-[1.75rem] border border-white/10 bg-slate-900/80">
                {selectedDocument?.mime_type?.includes("pdf") ? (
                  <iframe
                    title="Document viewer"
                    src={`${API_BASE_URL}/documents/${selectedDocument.id}/content#page=${viewerPage}`}
                    className="h-[560px] w-full bg-white"
                  />
                ) : selectedDocument ? (
                  <div className="flex h-[560px] flex-col justify-between p-8">
                    <div>
                      <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Document preview</p>
                      <h3 className="mt-3 text-3xl font-semibold text-white">{selectedDocument.name}</h3>
                      <p className="mt-4 max-w-xl text-sm leading-7 text-slate-300">
                        This file is indexed and searchable in the current workspace. PDF page navigation is available
                        for PDF assets, while structured files can still be cited and queried from the chat panel.
                      </p>
                    </div>
                    <div className="grid gap-3 md:grid-cols-2">
                      <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
                        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Chunks</p>
                        <p className="mt-2 text-3xl font-semibold text-white">{selectedDocument.chunk_count}</p>
                      </div>
                      <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
                        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Pages</p>
                        <p className="mt-2 text-3xl font-semibold text-white">{selectedDocument.page_count}</p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex h-[560px] items-center justify-center text-center text-slate-400">
                    Upload a document to unlock split-screen retrieval, citations, and page jumps.
                  </div>
                )}
              </div>

              <div className="space-y-4">
                <div className="rounded-[1.75rem] border border-cyan-300/15 bg-cyan-400/10 p-5">
                  <div className="flex items-center gap-3">
                    <DatabaseZap className="h-5 w-5 text-cyan-300" />
                    <p className="font-medium text-white">Workspace status</p>
                  </div>
                  <p className="mt-3 text-sm leading-7 text-cyan-50/80">{statusText}</p>
                </div>

                <div className="rounded-[1.75rem] border border-white/10 bg-white/5 p-5">
                  <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Active context</p>
                  <p className="mt-3 text-lg font-medium text-white">
                    {selectedWorkspace?.name || "No workspace selected"}
                  </p>
                  <p className="mt-2 text-sm leading-7 text-slate-300">
                    {selectedWorkspace?.description || "Create or select a workspace to group multiple sources together."}
                  </p>
                </div>

                <div className="rounded-[1.75rem] border border-white/10 bg-white/5 p-5">
                  <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Prompt ideas</p>
                  <div className="mt-4 flex flex-col gap-2">
                    {[
                      "Compare the main arguments across all uploaded sources.",
                      "Summarize the document and cite the strongest evidence.",
                      "Extract any numeric trends or structured observations.",
                    ].map((prompt) => (
                      <button
                        key={prompt}
                        type="button"
                        onClick={() => setMessageInput(prompt)}
                        className="rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-left text-sm text-slate-200 transition hover:bg-slate-900"
                      >
                        {prompt}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="flex min-h-[800px] flex-col rounded-[2rem] border border-white/10 bg-slate-950/55 p-5 shadow-2xl shadow-cyan-950/20 backdrop-blur-xl">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.35em] text-slate-400">Conversation</p>
                <h2 className="mt-2 text-2xl font-semibold text-white">Source-grounded chat</h2>
              </div>
              <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-300">
                <MessageSquareText className="h-4 w-4" />
                {selectedWorkspace?.name || "Waiting"}
              </div>
            </div>

            <div className="mt-5 flex-1 space-y-4 overflow-y-auto pr-2">
              {!messages.length && (
                <div className="rounded-[1.75rem] border border-dashed border-white/10 bg-white/5 p-8 text-center text-slate-400">
                  Start with a question once a workspace and document are selected.
                </div>
              )}

              {messages.map((message) => (
                <article
                  key={message.id}
                  className={`rounded-[1.75rem] border p-5 ${
                    message.role === "user"
                      ? "ml-12 border-cyan-400/20 bg-cyan-400/12 text-white"
                      : "mr-6 border-white/10 bg-white/5 text-slate-100"
                  }`}
                >
                  <div className="mb-4 flex items-center gap-3">
                    <div
                      className={`grid h-10 w-10 place-items-center rounded-2xl ${
                        message.role === "user" ? "bg-cyan-400 text-slate-950" : "bg-white/10 text-white"
                      }`}
                    >
                      {message.role === "user" ? <ArrowUp className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                    </div>
                    <div>
                      <p className="font-medium">{message.role === "user" ? "You" : "NexusRAG"}</p>
                      <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                        {message.streaming ? "Streaming" : "Ready"}
                      </p>
                    </div>
                  </div>

                  {message.role === "assistant" ? (
                    <div className="max-w-none text-sm leading-7 text-slate-200">
                      <MarkdownMessage>{message.content || "Thinking through the answer..."}</MarkdownMessage>
                    </div>
                  ) : (
                    <p className="whitespace-pre-wrap text-slate-50">{message.content}</p>
                  )}

                  {!!message.citations?.length && (
                    <div className="mt-5 flex flex-wrap gap-2">
                      {message.citations.map((citation, index) => (
                        <button
                          key={`${message.id}-${index}`}
                          type="button"
                          onClick={() => jumpToCitation(citation)}
                          className="rounded-full border border-sky-300/30 bg-sky-400/10 px-3 py-1.5 text-xs font-medium text-sky-100 transition hover:bg-sky-400/20"
                        >
                          Source {index + 1} • Page {citation.page_start}
                        </button>
                      ))}
                    </div>
                  )}
                </article>
              ))}
              <div ref={chatEndRef} />
            </div>

            <div className="mt-5 rounded-[1.75rem] border border-white/10 bg-slate-900/80 p-3">
              <div className="flex gap-3">
                <textarea
                  value={messageInput}
                  onChange={(event) => setMessageInput(event.target.value)}
                  onKeyDown={handleMessageKeyDown}
                  placeholder="Ask across the workspace, request a summary, or inspect a cited source..."
                  className="min-h-[92px] flex-1 resize-none rounded-[1.25rem] border border-white/10 bg-white/5 px-4 py-4 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-400/40 focus:bg-white/10"
                />
                <button
                  type="button"
                  onClick={() => void handleSendMessage()}
                  disabled={!messageInput.trim() || isStreaming}
                  className="inline-flex w-16 items-center justify-center rounded-[1.25rem] bg-cyan-400 text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isStreaming ? <LoaderCircle className="h-5 w-5 animate-spin" /> : <ArrowUp className="h-5 w-5" />}
                </button>
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
