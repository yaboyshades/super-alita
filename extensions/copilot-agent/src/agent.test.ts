import request from 'supertest';
import express from 'express';
import { initSSE, sendSSE, sendDefault } from '../src/sse';

describe('SSE Helper Functions', () => {
  let app: express.Application;
  let server: any;

  beforeEach(() => {
    app = express();
  });

  afterEach(() => {
    if (server) {
      server.close();
    }
  });

  test('initSSE sets correct headers', (done) => {
    app.get('/test-sse', (req, res) => {
      initSSE(res);
      res.end();
    });

    request(app)
      .get('/test-sse')
      .expect('Content-Type', 'text/event-stream; charset=utf-8')
      .expect('Cache-Control', 'no-cache, no-transform')
      .expect('Connection', 'keep-alive')
      .expect('X-Accel-Buffering', 'no')
      .expect(200, done);
  });

  test('sendSSE formats event correctly', (done) => {
    app.get('/test-event', (req, res) => {
      initSSE(res);
      sendSSE(res, 'copilot_confirmation', { id: 'test', title: 'Test Dialog' });
      res.end();
    });

    request(app)
      .get('/test-event')
      .expect(200)
      .expect((res) => {
        expect(res.text).toContain('event: copilot_confirmation\\n');
        expect(res.text).toContain('data: {"id":"test","title":"Test Dialog"}\\n\\n');
      })
      .end(done);
  });

  test('sendDefault formats data-only SSE', (done) => {
    app.get('/test-default', (req, res) => {
      initSSE(res);
      sendDefault(res, 'Hello from Super Alita');
      res.end();
    });

    request(app)
      .get('/test-default')
      .expect(200)
      .expect((res) => {
        expect(res.text).toContain('data: "Hello from Super Alita"\\n\\n');
        expect(res.text).not.toContain('event:');
      })
      .end(done);
  });
});

describe('gRPC Client Mock Functions', () => {
  test('getHealth returns valid health response', async () => {
    const { getHealth } = require('../src/grpcClient');
    const health = await getHealth();
    
    expect(health).toHaveProperty('status');
    expect(health).toHaveProperty('message');
    expect(health).toHaveProperty('timestamp');
    expect(typeof health.status).toBe('number');
    expect(typeof health.message).toBe('string');
  });

  test('getStatus returns system status', async () => {
    const { getStatus } = require('../src/grpcClient');
    const status = await getStatus();
    
    expect(status).toHaveProperty('cortex');
    expect(status).toHaveProperty('knowledge_graph');
    expect(status).toHaveProperty('optimization');
    expect(status).toHaveProperty('system');
    
    expect(status.cortex).toHaveProperty('active_sessions');
    expect(status.knowledge_graph).toHaveProperty('total_atoms');
    expect(status.optimization).toHaveProperty('active_policies');
    expect(status.system).toHaveProperty('components');
  });

  test('processTask handles task processing', async () => {
    const { processTask } = require('../src/grpcClient');
    const result = await processTask({
      task_id: 'test_123',
      content: 'analyze this data',
      session_id: 'session_456',
      user_id: 'user_789',
      workspace: 'workspace_abc',
      metadata: { source: 'test' }
    });
    
    expect(result).toHaveProperty('task_id', 'test_123');
    expect(result).toHaveProperty('status', 'completed');
    expect(result).toHaveProperty('result');
    expect(result).toHaveProperty('execution_time_ms');
    expect(typeof result.execution_time_ms).toBe('number');
  });

  test('kgQuery returns knowledge graph results', async () => {
    const { kgQuery } = require('../src/grpcClient');
    const result = await kgQuery({ query: 'machine learning', limit: 10 });
    
    expect(result).toHaveProperty('atoms');
    expect(result).toHaveProperty('bonds');
    expect(result).toHaveProperty('total_found');
    expect(Array.isArray(result.atoms)).toBe(true);
    expect(Array.isArray(result.bonds)).toBe(true);
  });

  test('banditDecide returns decision', async () => {
    const { banditDecide } = require('../src/grpcClient');
    const result = await banditDecide({ policy_id: 'exploration' });
    
    expect(result).toHaveProperty('decision_id');
    expect(result).toHaveProperty('algorithm');
    expect(result).toHaveProperty('action');
    expect(result).toHaveProperty('confidence');
    expect(result).toHaveProperty('expected_reward');
    expect(typeof result.confidence).toBe('number');
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });

  test('banditFeedback processes reward', async () => {
    const { banditFeedback } = require('../src/grpcClient');
    const result = await banditFeedback({
      decision_id: 'decision_123',
      reward: 0.8,
      source: 'test'
    });
    
    expect(result).toHaveProperty('success', true);
    expect(result).toHaveProperty('updated_policy');
    expect(result).toHaveProperty('new_confidence');
    expect(typeof result.new_confidence).toBe('number');
  });
});