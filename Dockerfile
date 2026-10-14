# NOTE: configuration only - not built/run in the development environment.
FROM python:3.12-slim
WORKDIR /app
COPY llm_gateway ./llm_gateway
ENV GATEWAY_HOST=0.0.0.0 GATEWAY_PORT=8080
EXPOSE 8080
CMD ["python", "-m", "llm_gateway.server"]
