# Build stage. Runs on the build host's own architecture and cross-compiles for
# the target, which is much faster than emulating the target under QEMU.
FROM --platform=$BUILDPLATFORM golang:1.24-alpine AS builder

ARG TARGETOS
ARG TARGETARCH

WORKDIR /app

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download all dependencies
RUN go mod download

# Copy the source code
COPY main.go .

# Build the application
RUN CGO_ENABLED=0 GOOS=${TARGETOS:-linux} GOARCH=${TARGETARCH} \
    go build -trimpath -ldflags="-s -w" -o /out/ollama-proxy .

# Final stage
FROM scratch

# Copy the binary from builder
COPY --from=builder /out/ollama-proxy /ollama-proxy

# Command to run
ENTRYPOINT ["/ollama-proxy"]
