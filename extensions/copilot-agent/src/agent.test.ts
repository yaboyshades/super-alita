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
        expect(res.text).toContain('event: copilot_confirmation');
        expect(res.text).toContain('data: {"id":"test","title":"Test Dialog"}');
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
        expect(res.text).toContain('data: "Hello from Super Alita"');
        expect(res.text).not.toContain('event:');
      })
      .end(done);
  });
});