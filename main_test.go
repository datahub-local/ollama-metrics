package main

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// The stub upstream reports these token counts for every request, whichever
// dialect was used, so tests can assert on exact counter deltas.
const (
	stubPromptTokens    = 11
	stubGeneratedTokens = 7
	stubEvalDurationNs  = 700000000
	stubStreamedContent = 2 // content chunks per streamed response
	stubFirstChunkDelay = 20 * time.Millisecond
	stubBetweenChunkGap = 10 * time.Millisecond
)

// recordedRequest is what the stub upstream saw, so tests can assert on how the
// proxy rewrote the request.
type recordedRequest struct {
	path          string
	body          []byte
	contentLength int64
	header        http.Header
}

type stubUpstream struct {
	*httptest.Server
	mu       sync.Mutex
	requests []recordedRequest
}

func (s *stubUpstream) lastRequest(t *testing.T, path string) recordedRequest {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := len(s.requests) - 1; i >= 0; i-- {
		if s.requests[i].path == path {
			return s.requests[i]
		}
	}
	t.Fatalf("upstream never received a request for %s", path)
	return recordedRequest{}
}

// sseEvents are the streamed chunks of an OpenAI-compatible response, in order.
// The usage chunk is the one Ollama only sends when include_usage is set.
func sseEvents(model string, includeUsage bool) []string {
	events := []string{
		fmt.Sprintf(`{"id":"1","object":"chat.completion.chunk","model":%q,"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}`, model),
		fmt.Sprintf(`{"id":"1","object":"chat.completion.chunk","model":%q,"choices":[{"index":0,"delta":{"content":"Hi"}}]}`, model),
		fmt.Sprintf(`{"id":"1","object":"chat.completion.chunk","model":%q,"choices":[{"index":0,"delta":{"content":" there"}}]}`, model),
		fmt.Sprintf(`{"id":"1","object":"chat.completion.chunk","model":%q,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`, model),
	}
	if includeUsage {
		events = append(events, fmt.Sprintf(
			`{"id":"1","object":"chat.completion.chunk","model":%q,"choices":[],"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}}`,
			model, stubPromptTokens, stubGeneratedTokens, stubPromptTokens+stubGeneratedTokens))
	}
	return append(events, "[DONE]")
}

// sseBody renders the events as they appear on the wire.
func sseBody(model string, includeUsage bool) string {
	var b strings.Builder
	for _, e := range sseEvents(model, includeUsage) {
		fmt.Fprintf(&b, "data: %s\n\n", e)
	}
	return b.String()
}

// nativeChatBody is the NDJSON stream of a native /api/chat response.
func nativeChatBody(model string) string {
	return strings.Join([]string{
		fmt.Sprintf(`{"model":%q,"message":{"role":"assistant","content":"Hi"},"done":false}`, model),
		fmt.Sprintf(`{"model":%q,"message":{"role":"assistant","content":" there"},"done":false}`, model),
		fmt.Sprintf(`{"model":%q,"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":%d,"eval_count":%d,"eval_duration":%d,"total_duration":1200000000}`,
			model, stubPromptTokens, stubGeneratedTokens, stubEvalDurationNs),
	}, "\n") + "\n"
}

func newStubUpstream(t *testing.T) *stubUpstream {
	t.Helper()
	stub := &stubUpstream{}

	mux := http.NewServeMux()

	mux.HandleFunc("/api/ps", func(w http.ResponseWriter, r *http.Request) {
		stub.record(r)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"models":[{"name":"stub-loaded","size":1048576,"processor":"cpu"}]}`)
	})

	mux.HandleFunc("/api/chat", func(w http.ResponseWriter, r *http.Request) {
		body := stub.record(r)
		model, _ := requestField(body, "model").(string)
		w.Header().Set("Content-Type", "application/x-ndjson")
		writeChunked(w, strings.SplitAfter(nativeChatBody(model), "\n"))
	})

	completions := func(w http.ResponseWriter, r *http.Request) {
		body := stub.record(r)
		model, _ := requestField(body, "model").(string)
		stream, _ := requestField(body, "stream").(bool)

		gzipReply := strings.Contains(r.Header.Get("Accept-Encoding"), "gzip")

		if !stream {
			var payload string
			if r.URL.Path == "/v1/completions" {
				payload = fmt.Sprintf(`{"id":"1","object":"text_completion","model":%q,"choices":[{"index":0,"text":"Hi there","finish_reason":"stop"}],"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}}`,
					model, stubPromptTokens, stubGeneratedTokens, stubPromptTokens+stubGeneratedTokens)
			} else {
				payload = fmt.Sprintf(`{"id":"1","object":"chat.completion","model":%q,"choices":[{"index":0,"message":{"role":"assistant","content":"Hi there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}}`,
					model, stubPromptTokens, stubGeneratedTokens, stubPromptTokens+stubGeneratedTokens)
			}
			w.Header().Set("Content-Type", "application/json")
			if gzipReply {
				// Mirrors real upstreams: compressed only because the request
				// asked for it. If the proxy forwards a client's
				// Accept-Encoding, Go stops decompressing for it and parsing
				// silently yields nothing.
				w.Header().Set("Content-Encoding", "gzip")
				zw := gzip.NewWriter(w)
				io.WriteString(zw, payload)
				zw.Close()
				return
			}
			io.WriteString(w, payload)
			return
		}

		includeUsage := false
		if opts, ok := requestField(body, "stream_options").(map[string]interface{}); ok {
			includeUsage, _ = opts["include_usage"].(bool)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		var events []string
		for _, e := range sseEvents(model, includeUsage) {
			events = append(events, fmt.Sprintf("data: %s\n\n", e))
		}
		writeChunked(w, events)
	}
	mux.HandleFunc("/v1/chat/completions", completions)
	mux.HandleFunc("/v1/completions", completions)

	mux.HandleFunc("/api/tags", func(w http.ResponseWriter, r *http.Request) {
		stub.record(r)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"models":[{"name":"stub-loaded","model":"stub-loaded","size_vram":1048576}]}`)
	})

	stub.Server = httptest.NewServer(mux)
	t.Cleanup(stub.Close)
	return stub
}

func (s *stubUpstream) record(r *http.Request) []byte {
	body, _ := io.ReadAll(r.Body)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests = append(s.requests, recordedRequest{
		path:          r.URL.Path,
		body:          body,
		contentLength: r.ContentLength,
		header:        r.Header.Clone(),
	})
	return body
}

// writeChunked writes each piece separately and flushes, so the proxy sees a
// real stream arriving over time rather than one buffered write.
func writeChunked(w http.ResponseWriter, pieces []string) {
	flusher, _ := w.(http.Flusher)
	first := true
	for _, p := range pieces {
		if p == "" {
			continue
		}
		if first {
			time.Sleep(stubFirstChunkDelay)
			first = false
		} else {
			time.Sleep(stubBetweenChunkGap)
		}
		io.WriteString(w, p)
		if flusher != nil {
			flusher.Flush()
		}
	}
}

func requestField(body []byte, key string) interface{} {
	var parsed map[string]interface{}
	if json.Unmarshal(body, &parsed) != nil {
		return nil
	}
	return parsed[key]
}

// newProxy starts the sidecar in front of a stub upstream.
func newProxy(t *testing.T) (*httptest.Server, *stubUpstream) {
	t.Helper()
	stub := newStubUpstream(t)
	proxy := httptest.NewServer(newMux(stub.URL))
	t.Cleanup(proxy.Close)
	return proxy, stub
}

// post sends a request through the proxy and returns the client-visible body.
func post(t *testing.T, proxy *httptest.Server, path, body string, header http.Header) string {
	t.Helper()
	got, _ := postFull(t, proxy, path, body, header)
	return got
}

// postFull also returns the client-visible response headers.
func postFull(t *testing.T, proxy *httptest.Server, path, body string, header http.Header) (string, http.Header) {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, proxy.URL+path, strings.NewReader(body))
	if err != nil {
		t.Fatalf("building request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	for name, vals := range header {
		for _, v := range vals {
			req.Header.Add(name, v)
		}
	}
	resp, err := proxy.Client().Do(req)
	if err != nil {
		t.Fatalf("POST %s: %v", path, err)
	}
	defer resp.Body.Close()
	got, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("reading response: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST %s: status %d, body %s", path, resp.StatusCode, got)
	}
	return string(got), resp.Header.Clone()
}

// scrape fetches the proxy's own /metrics output.
func scrape(t *testing.T, proxy *httptest.Server) string {
	t.Helper()
	resp, err := proxy.Client().Get(proxy.URL + "/metrics")
	if err != nil {
		t.Fatalf("GET /metrics: %v", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("reading /metrics: %v", err)
	}
	return string(body)
}

// sample returns the value of the first series whose name matches and whose
// label set contains every given label fragment (e.g. `model="x:latest"`).
func sample(t *testing.T, metrics, name string, labels ...string) (float64, bool) {
	t.Helper()
	for _, line := range strings.Split(metrics, "\n") {
		if !strings.HasPrefix(line, name+"{") {
			continue
		}
		matched := true
		for _, l := range labels {
			if !strings.Contains(line, l) {
				matched = false
				break
			}
		}
		if !matched {
			continue
		}
		fields := strings.Fields(line)
		v, err := strconv.ParseFloat(fields[len(fields)-1], 64)
		if err != nil {
			t.Fatalf("parsing value of %q: %v", line, err)
		}
		return v, true
	}
	return 0, false
}

func mustSample(t *testing.T, metrics, name string, labels ...string) float64 {
	t.Helper()
	v, ok := sample(t, metrics, name, labels...)
	if !ok {
		t.Fatalf("metric %s%v not found in /metrics output", name, labels)
	}
	return v
}

func requireAbsent(t *testing.T, metrics, name string, labels ...string) {
	t.Helper()
	if v, ok := sample(t, metrics, name, labels...); ok {
		t.Fatalf("metric %s%v unexpectedly present with value %v", name, labels, v)
	}
}

// TestNativeChatIsUnchanged pins the pre-existing /api/chat behaviour: the
// NDJSON stream reaches the client verbatim and the counts Ollama reports are
// recorded, now alongside a time-to-first-token observation.
func TestNativeChatIsUnchanged(t *testing.T) {
	proxy, _ := newProxy(t)
	const model = "native-chat"
	tagged := model + ":latest"

	got := post(t, proxy, "/api/chat", fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hi"}],"stream":true}`, model), nil)

	if want := nativeChatBody(model); got != want {
		t.Errorf("client saw a modified stream:\n got: %q\nwant: %q", got, want)
	}

	metrics := scrape(t, proxy)
	if v := mustSample(t, metrics, "ollama_prompt_tokens_total", `model="`+tagged+`"`); v != stubPromptTokens {
		t.Errorf("prompt tokens = %v, want %v", v, stubPromptTokens)
	}
	if v := mustSample(t, metrics, "ollama_generated_tokens_total", `model="`+tagged+`"`); v != stubGeneratedTokens {
		t.Errorf("generated tokens = %v, want %v", v, stubGeneratedTokens)
	}
	// The endpoint label is normalised and renamed to api_endpoint.
	if v := mustSample(t, metrics, "ollama_request_duration_seconds_count", `api_endpoint="chat"`, `model="`+tagged+`"`); v != 1 {
		t.Errorf("request duration count = %v, want 1", v)
	}
	// Ollama reports eval_duration itself, so seconds-per-token is exact.
	wantPerToken := float64(stubEvalDurationNs) / 1e9 / stubGeneratedTokens
	if v := mustSample(t, metrics, "ollama_time_per_token_seconds_sum", `model="`+tagged+`"`); v != wantPerToken {
		t.Errorf("time per token sum = %v, want %v", v, wantPerToken)
	}
	if v := mustSample(t, metrics, "ollama_time_to_first_token_seconds_count", `api_endpoint="chat"`, `model="`+tagged+`"`); v != 1 {
		t.Errorf("time to first token count = %v, want 1", v)
	}
}

// TestNativeChatNonStreaming checks that stream:false still yields token counts
// but no time-to-first-token, which only a stream can measure.
func TestNativeChatNonStreaming(t *testing.T) {
	proxy, _ := newProxy(t)
	const model = "native-chat-blocking"
	tagged := model + ":latest"

	post(t, proxy, "/api/chat", fmt.Sprintf(`{"model":%q,"messages":[],"stream":false}`, model), nil)

	metrics := scrape(t, proxy)
	if v := mustSample(t, metrics, "ollama_generated_tokens_total", `model="`+tagged+`"`); v != stubGeneratedTokens {
		t.Errorf("generated tokens = %v, want %v", v, stubGeneratedTokens)
	}
	requireAbsent(t, metrics, "ollama_time_to_first_token_seconds_count", `model="`+tagged+`"`)
}

// TestOpenAIStreamingInjectsAndHidesUsage is the core of the fix: usage is asked
// for on the client's behalf and the extra chunk never reaches the client.
func TestOpenAIStreamingInjectsAndHidesUsage(t *testing.T) {
	proxy, stub := newProxy(t)
	const model = "v1-stream-injected"
	tagged := model + ":latest"

	got := post(t, proxy, "/v1/chat/completions",
		fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":1000000}`, model), nil)

	// The forwarded request asked for usage even though the client did not.
	up := stub.lastRequest(t, "/v1/chat/completions")
	opts, ok := requestField(up.body, "stream_options").(map[string]interface{})
	if !ok {
		t.Fatalf("upstream request had no stream_options: %s", up.body)
	}
	if include, _ := opts["include_usage"].(bool); !include {
		t.Errorf("stream_options.include_usage not injected: %s", up.body)
	}
	// Injection must not mangle the rest of the body: re-marshalling through
	// interface{} would otherwise turn large integers into 1e+06, which Ollama's
	// integer fields reject.
	if !strings.Contains(string(up.body), `"max_tokens":1000000`) {
		t.Errorf("max_tokens literal was rewritten: %s", up.body)
	}
	if up.contentLength != int64(len(up.body)) {
		t.Errorf("upstream Content-Length = %d, want %d (the rewritten body length)", up.contentLength, len(up.body))
	}

	// The client sees exactly the stream it would have got without injection.
	if want := sseBody(model, false); got != want {
		t.Errorf("client saw the injected usage chunk:\n got: %q\nwant: %q", got, want)
	}
	if strings.Contains(got, "usage") {
		t.Errorf("client-visible stream mentions usage: %q", got)
	}

	metrics := scrape(t, proxy)
	if v := mustSample(t, metrics, "ollama_prompt_tokens_total", `model="`+tagged+`"`); v != stubPromptTokens {
		t.Errorf("prompt tokens = %v, want %v", v, stubPromptTokens)
	}
	if v := mustSample(t, metrics, "ollama_generated_tokens_total", `model="`+tagged+`"`); v != stubGeneratedTokens {
		t.Errorf("generated tokens = %v, want %v", v, stubGeneratedTokens)
	}
	if v := mustSample(t, metrics, "ollama_request_duration_seconds_count", `api_endpoint="v1/chat/completions"`, `model="`+tagged+`"`); v != 1 {
		t.Errorf("request duration count = %v, want 1", v)
	}
	if v := mustSample(t, metrics, "ollama_time_to_first_token_seconds_count", `api_endpoint="v1/chat/completions"`, `model="`+tagged+`"`); v != 1 {
		t.Errorf("time to first token count = %v, want 1", v)
	}
	// Measured across the generation window, so it must be well under the whole
	// request duration rather than derived from it.
	if v := mustSample(t, metrics, "ollama_time_per_token_seconds_count", `model="`+tagged+`"`); v != 1 {
		t.Errorf("time per token count = %v, want 1", v)
	}
	window := mustSample(t, metrics, "ollama_time_per_token_seconds_sum", `model="`+tagged+`"`)
	total := mustSample(t, metrics, "ollama_request_duration_seconds_sum", `api_endpoint="v1/chat/completions"`, `model="`+tagged+`"`)
	if window <= 0 {
		t.Errorf("time per token = %v, want a positive generation-window measurement", window)
	}
	if window*stubGeneratedTokens >= total {
		t.Errorf("time per token (%v) looks derived from total duration (%v)", window, total)
	}
}

// TestOpenAIStreamingKeepsClientRequestedUsage covers the other half: a client
// that asks for usage itself must still receive the chunk.
func TestOpenAIStreamingKeepsClientRequestedUsage(t *testing.T) {
	proxy, _ := newProxy(t)
	const model = "v1-stream-requested"
	tagged := model + ":latest"

	got := post(t, proxy, "/v1/chat/completions",
		fmt.Sprintf(`{"model":%q,"messages":[],"stream":true,"stream_options":{"include_usage":true}}`, model), nil)

	if want := sseBody(model, true); got != want {
		t.Errorf("client-requested usage chunk was not forwarded:\n got: %q\nwant: %q", got, want)
	}

	metrics := scrape(t, proxy)
	if v := mustSample(t, metrics, "ollama_generated_tokens_total", `model="`+tagged+`"`); v != stubGeneratedTokens {
		t.Errorf("generated tokens = %v, want %v", v, stubGeneratedTokens)
	}
}

// TestOpenAINonStreaming reads usage straight from the body, and deliberately
// records no seconds-per-token: with a large prompt and a short answer, prompt
// evaluation would dominate the request and skew the metric.
func TestOpenAINonStreaming(t *testing.T) {
	proxy, stub := newProxy(t)
	const model = "v1-blocking"
	tagged := model + ":latest"

	got := post(t, proxy, "/v1/chat/completions", fmt.Sprintf(`{"model":%q,"messages":[]}`, model), nil)
	if !strings.Contains(got, `"total_tokens":18`) {
		t.Errorf("non-streaming body was not forwarded intact: %q", got)
	}

	// Nothing to inject when the client is not streaming.
	up := stub.lastRequest(t, "/v1/chat/completions")
	if requestField(up.body, "stream_options") != nil {
		t.Errorf("stream_options injected into a non-streaming request: %s", up.body)
	}

	metrics := scrape(t, proxy)
	if v := mustSample(t, metrics, "ollama_prompt_tokens_total", `model="`+tagged+`"`); v != stubPromptTokens {
		t.Errorf("prompt tokens = %v, want %v", v, stubPromptTokens)
	}
	if v := mustSample(t, metrics, "ollama_request_duration_seconds_count", `api_endpoint="v1/chat/completions"`, `model="`+tagged+`"`); v != 1 {
		t.Errorf("request duration count = %v, want 1", v)
	}
	requireAbsent(t, metrics, "ollama_time_per_token_seconds_count", `model="`+tagged+`"`)
	requireAbsent(t, metrics, "ollama_time_to_first_token_seconds_count", `model="`+tagged+`"`)
}

// TestOpenAILegacyCompletions checks the second /v1 path, whose streamed content
// lives in choices[].text rather than a delta.
func TestOpenAILegacyCompletions(t *testing.T) {
	proxy, _ := newProxy(t)
	const model = "v1-legacy"
	tagged := model + ":latest"

	post(t, proxy, "/v1/completions", fmt.Sprintf(`{"model":%q,"prompt":"hi"}`, model), nil)

	metrics := scrape(t, proxy)
	if v := mustSample(t, metrics, "ollama_generated_tokens_total", `model="`+tagged+`"`); v != stubGeneratedTokens {
		t.Errorf("generated tokens = %v, want %v", v, stubGeneratedTokens)
	}
	if v := mustSample(t, metrics, "ollama_request_duration_seconds_count", `api_endpoint="v1/completions"`, `model="`+tagged+`"`); v != 1 {
		t.Errorf("request duration count = %v, want 1", v)
	}
}

// TestGzipRequestStillParses guards the Accept-Encoding bug: forwarding the
// client's header disables Go's transparent decompression, leaving the proxy
// parsing gzip bytes and silently counting nothing.
func TestGzipRequestStillParses(t *testing.T) {
	proxy, _ := newProxy(t)
	const model = "v1-gzip"
	tagged := model + ":latest"

	header := http.Header{"Accept-Encoding": []string{"gzip"}}
	got, respHeader := postFull(t, proxy, "/v1/chat/completions", fmt.Sprintf(`{"model":%q,"messages":[]}`, model), header)

	metrics := scrape(t, proxy)
	if v, ok := sample(t, metrics, "ollama_prompt_tokens_total", `model="`+tagged+`"`); !ok || v != stubPromptTokens {
		t.Errorf("prompt tokens = %v (present=%v), want %v: the upstream body arrived gzipped and parsed to nothing",
			v, ok, stubPromptTokens)
	}

	// Because the client's Accept-Encoding never reaches upstream, the body comes
	// back uncompressed and the client can read it without negotiating anything.
	if enc := respHeader.Get("Content-Encoding"); enc != "" {
		t.Errorf("response Content-Encoding = %q, want none", enc)
	}
	var decoded map[string]interface{}
	if err := json.Unmarshal([]byte(got), &decoded); err != nil {
		t.Errorf("client could not decode body %q: %v", got, err)
	}
}

// TestHistogramSeriesAppearOnlyAfterTraffic documents that the _bucket/_count/
// _sum series are missing until a request creates them: they are lazily created
// *Vec children, not a broken exporter.
func TestHistogramSeriesAppearOnlyAfterTraffic(t *testing.T) {
	proxy, _ := newProxy(t)
	const model = "v1-lazy-child"
	tagged := model + ":latest"
	labels := []string{`api_endpoint="v1/chat/completions"`, `model="` + tagged + `"`}

	before := scrape(t, proxy)
	for _, suffix := range []string{"_bucket", "_count", "_sum"} {
		requireAbsent(t, before, "ollama_request_duration_seconds"+suffix, labels...)
		requireAbsent(t, before, "ollama_time_to_first_token_seconds"+suffix, labels...)
	}

	post(t, proxy, "/v1/chat/completions", fmt.Sprintf(`{"model":%q,"messages":[],"stream":true}`, model), nil)

	after := scrape(t, proxy)
	for _, suffix := range []string{"_bucket", "_count", "_sum"} {
		mustSample(t, after, "ollama_request_duration_seconds"+suffix, labels...)
		mustSample(t, after, "ollama_time_to_first_token_seconds"+suffix, labels...)
		mustSample(t, after, "ollama_time_per_token_seconds"+suffix, `model="`+tagged+`"`)
	}
}

// TestLoadedModelGauges covers the /api/ps interception and the refresh the
// /metrics handler does on every scrape.
func TestLoadedModelGauges(t *testing.T) {
	proxy, _ := newProxy(t)

	resp, err := proxy.Client().Get(proxy.URL + "/api/ps")
	if err != nil {
		t.Fatalf("GET /api/ps: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !strings.Contains(string(body), "stub-loaded") {
		t.Errorf("/api/ps body not forwarded: %q", body)
	}

	metrics := scrape(t, proxy)
	if v := mustSample(t, metrics, "ollama_model_loaded", `model="stub-loaded:latest"`); v != 1 {
		t.Errorf("model_loaded = %v, want 1", v)
	}
	if v := mustSample(t, metrics, "ollama_model_ram_mb", `model="stub-loaded:latest"`); v != 1 {
		t.Errorf("model_ram_mb = %v, want 1", v)
	}
}

// TestPassthroughEndpoint checks an unparsed endpoint still gets a duration
// observation, labelled from the request body's model.
func TestPassthroughEndpoint(t *testing.T) {
	proxy, _ := newProxy(t)
	const model = "passthrough"
	tagged := model + ":latest"

	post(t, proxy, "/api/tags", fmt.Sprintf(`{"model":%q}`, model), nil)

	metrics := scrape(t, proxy)
	if v := mustSample(t, metrics, "ollama_request_duration_seconds_count", `api_endpoint="tags"`, `model="`+tagged+`"`); v != 1 {
		t.Errorf("request duration count = %v, want 1", v)
	}
}

func TestEndpointLabel(t *testing.T) {
	cases := map[string]string{
		"/api/chat":            "chat",
		"/api/generate":        "generate",
		"/api/ps":              "ps",
		"/v1/chat/completions": "v1/chat/completions",
		"/v1/completions":      "v1/completions",
		"/":                    "/",
	}
	for path, want := range cases {
		if got := endpointLabel(path); got != want {
			t.Errorf("endpointLabel(%q) = %q, want %q", path, got, want)
		}
	}
}

// TestInjectIncludeUsage covers the request-rewriting decision table on its own.
func TestInjectIncludeUsage(t *testing.T) {
	cases := []struct {
		name         string
		body         string
		wantInjected bool
	}{
		{"non-streaming is left alone", `{"model":"m"}`, false},
		{"explicit stream:false is left alone", `{"model":"m","stream":false}`, false},
		{"streaming gets usage injected", `{"model":"m","stream":true}`, true},
		{"empty stream_options gets usage injected", `{"model":"m","stream":true,"stream_options":{}}`, true},
		{"null stream_options gets usage injected", `{"model":"m","stream":true,"stream_options":null}`, true},
		{"client-requested usage is left alone", `{"model":"m","stream":true,"stream_options":{"include_usage":true}}`, false},
		{"unparseable body is left alone", `not json`, false},
		{"empty body is left alone", ``, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := []byte(tc.body)
			got, injected := injectIncludeUsage(parseRequestJSON(body), body)
			if injected != tc.wantInjected {
				t.Fatalf("injected = %v, want %v", injected, tc.wantInjected)
			}
			if !injected {
				if !bytes.Equal(got, body) {
					t.Errorf("body was rewritten to %q, want it untouched", got)
				}
				return
			}
			opts, ok := requestField(got, "stream_options").(map[string]interface{})
			if !ok {
				t.Fatalf("rewritten body has no stream_options object: %s", got)
			}
			if include, _ := opts["include_usage"].(bool); !include {
				t.Errorf("rewritten body does not set include_usage: %s", got)
			}
		})
	}
}
