import { emitDiagnostic, getFileInfo, readFileSnippet } from 'vscode:example/host-api';
import { emitMetric } from 'vscode:example/telemetry';

let dv = new DataView(new ArrayBuffer());
const dataView = mem => dv.buffer === mem.buffer ? dv : dv = new DataView(mem.buffer);

const toUint64 = val => BigInt.asUintN(64, BigInt(val));

function toUint32(val) {
  return val >>> 0;
}

const utf8Decoder = new TextDecoder();

const utf8Encoder = new TextEncoder();
let utf8EncodedLen = 0;
function utf8Encode(s, realloc, memory) {
  if (typeof s !== 'string') throw new TypeError('expected a string');
  if (s.length === 0) {
    utf8EncodedLen = 0;
    return 1;
  }
  let buf = utf8Encoder.encode(s);
  let ptr = realloc(0, 0, 1, buf.length);
  new Uint8Array(memory.buffer).set(buf, ptr);
  utf8EncodedLen = buf.length;
  return ptr;
}

let NEXT_TASK_ID = 0n;
function startCurrentTask(componentIdx, isAsync, entryFnName) {
  _debugLog('[startCurrentTask()] args', { componentIdx, isAsync });
  if (componentIdx === undefined || componentIdx === null) {
    throw new Error('missing/invalid component instance index while starting task');
  }
  const tasks = ASYNC_TASKS_BY_COMPONENT_IDX.get(componentIdx);
  
  const nextId = ++NEXT_TASK_ID;
  const newTask = new AsyncTask({ id: nextId, componentIdx, isAsync, entryFnName });
  const newTaskMeta = { id: nextId, componentIdx, task: newTask };
  
  ASYNC_CURRENT_TASK_IDS.push(nextId);
  ASYNC_CURRENT_COMPONENT_IDXS.push(componentIdx);
  
  if (!tasks) {
    ASYNC_TASKS_BY_COMPONENT_IDX.set(componentIdx, [newTaskMeta]);
    return nextId;
  } else {
    tasks.push(newTaskMeta);
  }
  
  return nextId;
}

function endCurrentTask(componentIdx, taskId) {
  _debugLog('[endCurrentTask()] args', { componentIdx });
  componentIdx ??= ASYNC_CURRENT_COMPONENT_IDXS.at(-1);
  taskId ??= ASYNC_CURRENT_TASK_IDS.at(-1);
  if (componentIdx === undefined || componentIdx === null) {
    throw new Error('missing/invalid component instance index while ending current task');
  }
  const tasks = ASYNC_TASKS_BY_COMPONENT_IDX.get(componentIdx);
  if (!tasks || !Array.isArray(tasks)) {
    throw new Error('missing/invalid tasks for component instance while ending task');
  }
  if (tasks.length == 0) {
    throw new Error('no current task(s) for component instance while ending task');
  }
  
  if (taskId) {
    const last = tasks[tasks.length - 1];
    if (last.id !== taskId) {
      throw new Error('current task does not match expected task ID');
    }
  }
  
  ASYNC_CURRENT_TASK_IDS.pop();
  ASYNC_CURRENT_COMPONENT_IDXS.pop();
  
  return tasks.pop();
}
const ASYNC_TASKS_BY_COMPONENT_IDX = new Map();
const ASYNC_CURRENT_TASK_IDS = [];
const ASYNC_CURRENT_COMPONENT_IDXS = [];

class AsyncTask {
  static State = {
    INITIAL: 'initial',
    CANCELLED: 'cancelled',
    CANCEL_PENDING: 'cancel-pending',
    CANCEL_DELIVERED: 'cancel-delivered',
    RESOLVED: 'resolved',
  }
  
  static BlockResult = {
    CANCELLED: 'block.cancelled',
    NOT_CANCELLED: 'block.not-cancelled',
  }
  
  #id;
  #componentIdx;
  #state;
  #isAsync;
  #onResolve = null;
  #returnedResults = null;
  #entryFnName = null;
  
  cancelled = false;
  requested = false;
  alwaysTaskReturn = false;
  
  returnCalls =  0;
  storage = [0, 0];
  borrowedHandles = {};
  
  awaitableResume = null;
  awaitableCancel = null;
  
  constructor(opts) {
    if (opts?.id === undefined) { throw new TypeError('missing task ID during task creation'); }
    this.#id = opts.id;
    if (opts?.componentIdx === undefined) {
      throw new TypeError('missing component id during task creation');
    }
    this.#componentIdx = opts.componentIdx;
    this.#state = AsyncTask.State.INITIAL;
    this.#isAsync = opts?.isAsync ?? false;
    this.#entryFnName = opts.entryFnName;
    
    this.#onResolve = (results) => {
      this.#returnedResults = results;
    }
  }
  
  taskState() { return this.#state.slice(); }
  id() { return this.#id; }
  componentIdx() { return this.#componentIdx; }
  isAsync() { return this.#isAsync; }
  getEntryFnName() { return this.#entryFnName; }
  
  takeResults() {
    const results = this.#returnedResults;
    this.#returnedResults = null;
    return results;
  }
  
  mayEnter(task) {
    const cstate = getOrCreateAsyncState(this.#componentIdx);
    if (!cstate.backpressure) {
      _debugLog('[AsyncTask#mayEnter()] disallowed due to backpressure', { taskID: this.#id });
      return false;
    }
    if (!cstate.callingSyncImport()) {
      _debugLog('[AsyncTask#mayEnter()] disallowed due to sync import call', { taskID: this.#id });
      return false;
    }
    const callingSyncExportWithSyncPending = cstate.callingSyncExport && !task.isAsync;
    if (!callingSyncExportWithSyncPending) {
      _debugLog('[AsyncTask#mayEnter()] disallowed due to sync export w/ sync pending', { taskID: this.#id });
      return false;
    }
    return true;
  }
  
  async enter() {
    _debugLog('[AsyncTask#enter()] args', { taskID: this.#id });
    
    // TODO: assert scheduler locked
    // TODO: trap if on the stack
    
    const cstate = getOrCreateAsyncState(this.#componentIdx);
    
    let mayNotEnter = !this.mayEnter(this);
    const componentHasPendingTasks = cstate.pendingTasks > 0;
    if (mayNotEnter || componentHasPendingTasks) {
      
      throw new Error('in enter()'); // TODO: remove
      cstate.pendingTasks.set(this.#id, new Awaitable(new Promise()));
      
      const blockResult = await this.onBlock(awaitable);
      if (blockResult) {
        // TODO: find this pending task in the component
        const pendingTask = cstate.pendingTasks.get(this.#id);
        if (!pendingTask) {
          throw new Error('pending task [' + this.#id + '] not found for component instance');
        }
        cstate.pendingTasks.remove(this.#id);
        this.#onResolve([]);
        return false;
      }
      
      mayNotEnter = !this.mayEnter(this);
      if (!mayNotEnter || !cstate.startPendingTask) {
        throw new Error('invalid component entrance/pending task resolution');
      }
      cstate.startPendingTask = false;
    }
    
    if (!this.isAsync) { cstate.callingSyncExport = true; }
    
    return true;
  }
  
  async waitForEvent(opts) {
    const { waitableSetRep, isAsync } = opts;
    _debugLog('[AsyncTask#waitForEvent()] args', { taskID: this.#id, waitableSetRep, isAsync });
    
    if (this.#isAsync !== isAsync) {
      throw new Error('async waitForEvent called on non-async task');
    }
    
    if (this.status === AsyncTask.State.CANCEL_PENDING) {
      this.#state = AsyncTask.State.CANCEL_DELIVERED;
      return {
        code: ASYNC_EVENT_CODE.TASK_CANCELLED,
        something: 0,
        something: 0,
      };
    }
    
    const state = getOrCreateAsyncState(this.#componentIdx);
    const waitableSet = state.waitableSets.get(waitableSetRep);
    if (!waitableSet) { throw new Error('missing/invalid waitable set'); }
    
    waitableSet.numWaiting += 1;
    let event = null;
    
    while (event == null) {
      const awaitable = new Awaitable(waitableSet.getPendingEvent());
      const waited = await this.blockOn({ awaitable, isAsync, isCancellable: true });
      if (waited) {
        if (this.#state !== AsyncTask.State.INITIAL) {
          throw new Error('task should be in initial state found [' + this.#state + ']');
        }
        this.#state = AsyncTask.State.CANCELLED;
        return {
          code: ASYNC_EVENT_CODE.TASK_CANCELLED,
          something: 0,
          something: 0,
        };
      }
      
      event = waitableSet.poll();
    }
    
    waitableSet.numWaiting -= 1;
    return event;
  }
  
  waitForEventSync(opts) {
    throw new Error('AsyncTask#yieldSync() not implemented')
  }
  
  async pollForEvent(opts) {
    const { waitableSetRep, isAsync } = opts;
    _debugLog('[AsyncTask#pollForEvent()] args', { taskID: this.#id, waitableSetRep, isAsync });
    
    if (this.#isAsync !== isAsync) {
      throw new Error('async pollForEvent called on non-async task');
    }
    
    throw new Error('AsyncTask#pollForEvent() not implemented');
  }
  
  pollForEventSync(opts) {
    throw new Error('AsyncTask#yieldSync() not implemented')
  }
  
  async blockOn(opts) {
    const { awaitable, isCancellable, forCallback } = opts;
    _debugLog('[AsyncTask#blockOn()] args', { taskID: this.#id, awaitable, isCancellable, forCallback });
    
    if (awaitable.resolved() && !ASYNC_DETERMINISM && _coinFlip()) {
      return AsyncTask.BlockResult.NOT_CANCELLED;
    }
    
    const cstate = getOrCreateAsyncState(this.#componentIdx);
    if (forCallback) { cstate.exclusiveRelease(); }
    
    let cancelled = await this.onBlock(awaitable);
    if (cancelled === AsyncTask.BlockResult.CANCELLED && !isCancellable) {
      const secondCancel = await this.onBlock(awaitable);
      if (secondCancel !== AsyncTask.BlockResult.NOT_CANCELLED) {
        throw new Error('uncancellable task was canceled despite second onBlock()');
      }
    }
    
    if (forCallback) {
      const acquired = new Awaitable(cstate.exclusiveLock());
      cancelled = await this.onBlock(acquired);
      if (cancelled === AsyncTask.BlockResult.CANCELLED) {
        const secondCancel = await this.onBlock(acquired);
        if (secondCancel !== AsyncTask.BlockResult.NOT_CANCELLED) {
          throw new Error('uncancellable callback task was canceled despite second onBlock()');
        }
      }
    }
    
    if (cancelled === AsyncTask.BlockResult.CANCELLED) {
      if (this.#state !== AsyncTask.State.INITIAL) {
        throw new Error('cancelled task is not at initial state');
      }
      if (isCancellable) {
        this.#state = AsyncTask.State.CANCELLED;
        return AsyncTask.BlockResult.CANCELLED;
      } else {
        this.#state = AsyncTask.State.CANCEL_PENDING;
        return AsyncTask.BlockResult.NOT_CANCELLED;
      }
    }
    
    return AsyncTask.BlockResult.NOT_CANCELLED;
  }
  
  async onBlock(awaitable) {
    _debugLog('[AsyncTask#onBlock()] args', { taskID: this.#id, awaitable });
    if (!(awaitable instanceof Awaitable)) {
      throw new Error('invalid awaitable during onBlock');
    }
    
    // Build a promise that this task can await on which resolves when it is awoken
    const { promise, resolve, reject } = Promise.withResolvers();
    this.awaitableResume = () => {
      _debugLog('[AsyncTask] resuming after onBlock', { taskID: this.#id });
      resolve();
    };
    this.awaitableCancel = (err) => {
      _debugLog('[AsyncTask] rejecting after onBlock', { taskID: this.#id, err });
      reject(err);
    };
    
    // Park this task/execution to be handled later
    const state = getOrCreateAsyncState(this.#componentIdx);
    state.parkTaskOnAwaitable({ awaitable, task: this });
    
    try {
      await promise;
      return AsyncTask.BlockResult.NOT_CANCELLED;
    } catch (err) {
      // rejection means task cancellation
      return AsyncTask.BlockResult.CANCELLED;
    }
  }
  
  // NOTE: this should likely be moved to a SubTask class
  async asyncOnBlock(awaitable) {
    _debugLog('[AsyncTask#asyncOnBlock()] args', { taskID: this.#id, awaitable });
    if (!(awaitable instanceof Awaitable)) {
      throw new Error('invalid awaitable during onBlock');
    }
    // TODO: watch for waitable AND cancellation
    // TODO: if it WAS cancelled:
    // - return true
    // - only once per subtask
    // - do not wait on the scheduler
    // - control flow should go to the subtask (only once)
    // - Once subtask blocks/resolves, reqlinquishControl() will tehn resolve request_cancel_end (without scheduler lock release)
    // - control flow goes back to request_cancel
    //
    // Subtask cancellation should work similarly to an async import call -- runs sync up until
    // the subtask blocks or resolves
    //
    throw new Error('AsyncTask#asyncOnBlock() not yet implemented');
  }
  
  async yield(opts) {
    const { isCancellable, forCallback } = opts;
    _debugLog('[AsyncTask#yield()] args', { taskID: this.#id, isCancellable, forCallback });
    
    if (isCancellable && this.status === AsyncTask.State.CANCEL_PENDING) {
      this.#state = AsyncTask.State.CANCELLED;
      return {
        code: ASYNC_EVENT_CODE.TASK_CANCELLED,
        payload: [0, 0],
      };
    }
    
    // TODO: Awaitables need to *always* trigger the parking mechanism when they're done...?
    // TODO: Component async state should remember which awaitables are done and work to clear tasks waiting
    
    const blockResult = await this.blockOn({
      awaitable: new Awaitable(new Promise(resolve => setTimeout(resolve, 0))),
      isCancellable,
      forCallback,
    });
    
    if (blockResult === AsyncTask.BlockResult.CANCELLED) {
      if (this.#state !== AsyncTask.State.INITIAL) {
        throw new Error('task should be in initial state found [' + this.#state + ']');
      }
      this.#state = AsyncTask.State.CANCELLED;
      return {
        code: ASYNC_EVENT_CODE.TASK_CANCELLED,
        payload: [0, 0],
      };
    }
    
    return {
      code: ASYNC_EVENT_CODE.NONE,
      payload: [0, 0],
    };
  }
  
  yieldSync(opts) {
    throw new Error('AsyncTask#yieldSync() not implemented')
  }
  
  cancel() {
    _debugLog('[AsyncTask#cancel()] args', { });
    if (!this.taskState() !== AsyncTask.State.CANCEL_DELIVERED) {
      throw new Error('invalid task state for cancellation');
    }
    if (this.borrowedHandles.length > 0) { throw new Error('task still has borrow handles'); }
    
    this.#onResolve([]);
    this.#state = AsyncTask.State.RESOLVED;
  }
  
  resolve(result) {
    if (this.#state === AsyncTask.State.RESOLVED) {
      throw new Error('task is already resolved');
    }
    if (this.borrowedHandles.length > 0) { throw new Error('task still has borrow handles'); }
    this.#onResolve(result);
    this.#state = AsyncTask.State.RESOLVED;
  }
  
  exit() {
    // TODO: ensure there is only one task at a time (scheduler.lock() functionality)
    if (this.#state !== AsyncTask.State.RESOLVED) {
      throw new Error('task exited without resolution');
    }
    if (this.borrowedHandles > 0) {
      throw new Error('task exited without clearing borrowed handles');
    }
    
    const state = getOrCreateAsyncState(this.#componentIdx);
    if (!state) { throw new Error('missing async state for component [' + this.#componentIdx + ']'); }
    if (!this.#isAsync && !state.inSyncExportCall) {
      throw new Error('sync task must be run from components known to be in a sync export call');
    }
    state.inSyncExportCall = false;
    
    this.startPendingTask();
  }
  
  startPendingTask(opts) {
    // TODO: implement
  }
  
}

function unpackCallbackResult(result) {
  _debugLog('[unpackCallbackResult()] args', { result });
  if (!(_typeCheckValidI32(result))) { throw new Error('invalid callback return value [' + result + '], not a valid i32'); }
  const eventCode = result & 0xF;
  if (eventCode < 0 || eventCode > 3) {
    throw new Error('invalid async return value [' + eventCode + '], outside callback code range');
  }
  if (result < 0 || result >= 2**32) { throw new Error('invalid callback result'); }
  // TODO: table max length check?
  const waitableSetIdx = result >> 4;
  return [eventCode, waitableSetIdx];
}
const ASYNC_STATE = new Map();

function getOrCreateAsyncState(componentIdx, init) {
  if (!ASYNC_STATE.has(componentIdx)) {
    ASYNC_STATE.set(componentIdx, new ComponentAsyncState());
  }
  return ASYNC_STATE.get(componentIdx);
}

class ComponentAsyncState {
  #callingAsyncImport = false;
  #syncImportWait = Promise.withResolvers();
  #lock = null;
  
  mayLeave = false;
  waitableSets = new RepTable();
  waitables = new RepTable();
  
  #parkedTasks = new Map();
  
  callingSyncImport(val) {
    if (val === undefined) { return this.#callingAsyncImport; }
    if (typeof val !== 'boolean') { throw new TypeError('invalid setting for async import'); }
    const prev = this.#callingAsyncImport;
    this.#callingAsyncImport = val;
    if (prev === true && this.#callingAsyncImport === false) {
      this.#notifySyncImportEnd();
    }
  }
  
  #notifySyncImportEnd() {
    const existing = this.#syncImportWait;
    this.#syncImportWait = Promise.withResolvers();
    existing.resolve();
  }
  
  async waitForSyncImportCallEnd() {
    await this.#syncImportWait.promise;
  }
  
  parkTaskOnAwaitable(args) {
    if (!args.awaitable) { throw new TypeError('missing awaitable when trying to park'); }
    if (!args.task) { throw new TypeError('missing task when trying to park'); }
    const { awaitable, task } = args;
    
    let taskList = this.#parkedTasks.get(awaitable.id());
    if (!taskList) {
      taskList = [];
      this.#parkedTasks.set(awaitable.id(), taskList);
    }
    taskList.push(task);
    
    this.wakeNextTaskForAwaitable(awaitable);
  }
  
  wakeNextTaskForAwaitable(awaitable) {
    if (!awaitable) { throw new TypeError('missing awaitable when waking next task'); }
    const awaitableID = awaitable.id();
    
    const taskList = this.#parkedTasks.get(awaitableID);
    if (!taskList || taskList.length === 0) {
      _debugLog('[ComponentAsyncState] no tasks waiting for awaitable', { awaitableID: awaitable.id() });
      return;
    }
    
    let task = taskList.shift(); // todo(perf)
    if (!task) { throw new Error('no task in parked list despite previous check'); }
    
    if (!task.awaitableResume) {
      throw new Error('task ready due to awaitable is missing resume', { taskID: task.id(), awaitableID });
    }
    task.awaitableResume();
  }
  
  async exclusiveLock() {  // TODO: use atomics
  if (this.#lock === null) {
    this.#lock = { ticket: 0n };
  }
  
  // Take a ticket for the next valid usage
  const ticket = ++this.#lock.ticket;
  
  _debugLog('[ComponentAsyncState#exclusiveLock()] locking', {
    currentTicket: ticket - 1n,
    ticket
  });
  
  // If there is an active promise, then wait for it
  let finishedTicket;
  while (this.#lock.promise) {
    finishedTicket = await this.#lock.promise;
    if (finishedTicket === ticket - 1n) { break; }
  }
  
  const { promise, resolve } = Promise.withResolvers();
  this.#lock = {
    ticket,
    promise,
    resolve,
  };
  
  return this.#lock.promise;
}

exclusiveRelease() {
  _debugLog('[ComponentAsyncState#exclusiveRelease()] releasing', {
    currentTicket: this.#lock === null ? 'none' : this.#lock.ticket,
  });
  
  if (this.#lock === null) { return; }
  
  const existingLock = this.#lock;
  this.#lock = null;
  existingLock.resolve(existingLock.ticket);
}

isExclusivelyLocked() { return this.#lock !== null; }

}

if (!Promise.withResolvers) {
  Promise.withResolvers = () => {
    let resolve;
    let reject;
    const promise = new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    });
    return { promise, resolve, reject };
  };
}

const _debugLog = (...args) => {
  if (!globalThis?.process?.env?.JCO_DEBUG) { return; }
  console.debug(...args);
}
const ASYNC_DETERMINISM = 'random';
const _coinFlip = () => { return Math.random() > 0.5; };
const I32_MAX = 2_147_483_647;
const I32_MIN = -2_147_483_648;
const _typeCheckValidI32 = (n) => typeof n === 'number' && n >= I32_MIN && n <= I32_MAX;

const base64Compile = str => WebAssembly.compile(typeof Buffer !== 'undefined' ? Buffer.from(str, 'base64') : Uint8Array.from(atob(str), b => b.charCodeAt(0)));

function clampGuest(i, min, max) {
  if (i < min || i > max) throw new TypeError(`must be between ${min} and ${max}`);
  return i;
}

class ComponentError extends Error {
  constructor (value) {
    const enumerable = typeof value !== 'string';
    super(enumerable ? `${String(value)} (see error.payload)` : value);
    Object.defineProperty(this, 'payload', { value, enumerable });
  }
}

function getErrorPayload(e) {
  if (e && hasOwnProperty.call(e, 'payload')) return e.payload;
  if (e instanceof Error) throw e;
  return e;
}

class RepTable {
  #data = [0, null];
  
  insert(val) {
    _debugLog('[RepTable#insert()] args', { val });
    const freeIdx = this.#data[0];
    if (freeIdx === 0) {
      this.#data.push(val);
      this.#data.push(null);
      return (this.#data.length >> 1) - 1;
    }
    this.#data[0] = this.#data[freeIdx];
    const newFreeIdx = freeIdx << 1;
    this.#data[newFreeIdx] = val;
    this.#data[newFreeIdx + 1] = null;
    return free;
  }
  
  get(rep) {
    _debugLog('[RepTable#insert()] args', { rep });
    const baseIdx = idx << 1;
    const val = this.#data[baseIdx];
    return val;
  }
  
  contains(rep) {
    _debugLog('[RepTable#insert()] args', { rep });
    const baseIdx = idx << 1;
    return !!this.#data[baseIdx];
  }
  
  remove(rep) {
    _debugLog('[RepTable#insert()] args', { idx });
    if (this.#data.length === 2) { throw new Error('invalid'); }
    
    const baseIdx = idx << 1;
    const val = this.#data[baseIdx];
    if (val === 0) { throw new Error('invalid resource rep (cannot be 0)'); }
    this.#data[baseIdx] = this.#data[0];
    this.#data[0] = idx;
    return val;
  }
  
  clear() {
    this.#data = [0, null];
  }
}

const hasOwnProperty = Object.prototype.hasOwnProperty;

const instantiateCore = WebAssembly.instantiate;


let exports0;
let exports1;
let memory0;
let realloc0;

function trampoline0(arg0, arg1, arg2, arg3, arg4) {
  var ptr0 = arg0;
  var len0 = arg1;
  var result0 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr0, len0));
  _debugLog('[iface="vscode:example/telemetry", function="emit-metric"] [Instruction::CallInterface] (async? sync, @ enter)');
  const _interface_call_currentTaskID = startCurrentTask(0, false, 'emit-metric');
  emitMetric({
    operation: result0,
    durationMs: arg2 >>> 0,
    memoryUsed: arg3 >>> 0,
    timestamp: BigInt.asUintN(64, arg4),
  });
  _debugLog('[iface="vscode:example/telemetry", function="emit-metric"] [Instruction::CallInterface] (sync, @ post-call)');
  endCurrentTask(0);
  _debugLog('[iface="vscode:example/telemetry", function="emit-metric"][Instruction::Return]', {
    funcName: 'emit-metric',
    paramCount: 0,
    postReturn: false
  });
}


function trampoline1(arg0, arg1, arg2) {
  var ptr0 = arg0;
  var len0 = arg1;
  var result0 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr0, len0));
  _debugLog('[iface="vscode:example/host-api", function="get-file-info"] [Instruction::CallInterface] (async? sync, @ enter)');
  const _interface_call_currentTaskID = startCurrentTask(0, false, 'get-file-info');
  let ret;
  try {
    ret = { tag: 'ok', val: getFileInfo(result0)};
  } catch (e) {
    ret = { tag: 'err', val: getErrorPayload(e) };
  }
  _debugLog('[iface="vscode:example/host-api", function="get-file-info"] [Instruction::CallInterface] (sync, @ post-call)');
  endCurrentTask(0);
  var variant4 = ret;
  switch (variant4.tag) {
    case 'ok': {
      const e = variant4.val;
      dataView(memory0).setInt8(arg2 + 0, 0, true);
      var {path: v1_0, size: v1_1, modified: v1_2 } = e;
      var ptr2 = utf8Encode(v1_0, realloc0, memory0);
      var len2 = utf8EncodedLen;
      dataView(memory0).setUint32(arg2 + 12, len2, true);
      dataView(memory0).setUint32(arg2 + 8, ptr2, true);
      dataView(memory0).setInt32(arg2 + 16, toUint32(v1_1), true);
      dataView(memory0).setBigInt64(arg2 + 24, toUint64(v1_2), true);
      break;
    }
    case 'err': {
      const e = variant4.val;
      dataView(memory0).setInt8(arg2 + 0, 1, true);
      var ptr3 = utf8Encode(e, realloc0, memory0);
      var len3 = utf8EncodedLen;
      dataView(memory0).setUint32(arg2 + 12, len3, true);
      dataView(memory0).setUint32(arg2 + 8, ptr3, true);
      break;
    }
    default: {
      throw new TypeError('invalid variant specified for result');
    }
  }
  _debugLog('[iface="vscode:example/host-api", function="get-file-info"][Instruction::Return]', {
    funcName: 'get-file-info',
    paramCount: 0,
    postReturn: false
  });
}


function trampoline2(arg0, arg1, arg2, arg3, arg4) {
  var ptr0 = arg0;
  var len0 = arg1;
  var result0 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr0, len0));
  _debugLog('[iface="vscode:example/host-api", function="read-file-snippet"] [Instruction::CallInterface] (async? sync, @ enter)');
  const _interface_call_currentTaskID = startCurrentTask(0, false, 'read-file-snippet');
  let ret;
  try {
    ret = { tag: 'ok', val: readFileSnippet(result0, arg2 >>> 0, arg3 >>> 0)};
  } catch (e) {
    ret = { tag: 'err', val: getErrorPayload(e) };
  }
  _debugLog('[iface="vscode:example/host-api", function="read-file-snippet"] [Instruction::CallInterface] (sync, @ post-call)');
  endCurrentTask(0);
  var variant3 = ret;
  switch (variant3.tag) {
    case 'ok': {
      const e = variant3.val;
      dataView(memory0).setInt8(arg4 + 0, 0, true);
      var ptr1 = utf8Encode(e, realloc0, memory0);
      var len1 = utf8EncodedLen;
      dataView(memory0).setUint32(arg4 + 8, len1, true);
      dataView(memory0).setUint32(arg4 + 4, ptr1, true);
      break;
    }
    case 'err': {
      const e = variant3.val;
      dataView(memory0).setInt8(arg4 + 0, 1, true);
      var ptr2 = utf8Encode(e, realloc0, memory0);
      var len2 = utf8EncodedLen;
      dataView(memory0).setUint32(arg4 + 8, len2, true);
      dataView(memory0).setUint32(arg4 + 4, ptr2, true);
      break;
    }
    default: {
      throw new TypeError('invalid variant specified for result');
    }
  }
  _debugLog('[iface="vscode:example/host-api", function="read-file-snippet"][Instruction::Return]', {
    funcName: 'read-file-snippet',
    paramCount: 0,
    postReturn: false
  });
}


function trampoline3(arg0, arg1, arg2, arg3, arg4) {
  var ptr0 = arg0;
  var len0 = arg1;
  var result0 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr0, len0));
  var ptr1 = arg3;
  var len1 = arg4;
  var result1 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr1, len1));
  _debugLog('[iface="vscode:example/host-api", function="emit-diagnostic"] [Instruction::CallInterface] (async? sync, @ enter)');
  const _interface_call_currentTaskID = startCurrentTask(0, false, 'emit-diagnostic');
  emitDiagnostic(result0, arg2 >>> 0, result1);
  _debugLog('[iface="vscode:example/host-api", function="emit-diagnostic"] [Instruction::CallInterface] (sync, @ post-call)');
  endCurrentTask(0);
  _debugLog('[iface="vscode:example/host-api", function="emit-diagnostic"][Instruction::Return]', {
    funcName: 'emit-diagnostic',
    paramCount: 0,
    postReturn: false
  });
}

let exports2;
let exports3;
let postReturn0;
let postReturn1;
let postReturn2;
let postReturn3;
let postReturn4;
let exports1Cm32p2Analyze;

function analyze(arg0) {
  var ptr0 = utf8Encode(arg0, realloc0, memory0);
  var len0 = utf8EncodedLen;
  _debugLog('[iface="analyze", function="analyze"] [Instruction::CallWasm] (async? false, @ enter)');
  const _wasm_call_currentTaskID = startCurrentTask(0, false, 'exports1Cm32p2Analyze');
  const ret = exports1Cm32p2Analyze(ptr0, len0);
  endCurrentTask(0);
  var len5 = dataView(memory0).getUint32(ret + 4, true);
  var base5 = dataView(memory0).getUint32(ret + 0, true);
  var result5 = [];
  for (let i = 0; i < len5; i++) {
    const base = base5 + i * 40;
    var ptr1 = dataView(memory0).getUint32(base + 12, true);
    var len1 = dataView(memory0).getUint32(base + 16, true);
    var result1 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr1, len1));
    var ptr2 = dataView(memory0).getUint32(base + 20, true);
    var len2 = dataView(memory0).getUint32(base + 24, true);
    var result2 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr2, len2));
    let variant4;
    switch (dataView(memory0).getUint8(base + 28, true)) {
      case 0: {
        variant4 = undefined;
        break;
      }
      case 1: {
        var ptr3 = dataView(memory0).getUint32(base + 32, true);
        var len3 = dataView(memory0).getUint32(base + 36, true);
        var result3 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr3, len3));
        variant4 = result3;
        break;
      }
      default: {
        throw new TypeError('invalid variant discriminant for option');
      }
    }
    result5.push({
      line: dataView(memory0).getInt32(base + 0, true) >>> 0,
      col: dataView(memory0).getInt32(base + 4, true) >>> 0,
      severity: clampGuest(dataView(memory0).getUint8(base + 8, true), 0, 255),
      code: result1,
      message: result2,
      suggestion: variant4,
    });
  }
  _debugLog('[iface="analyze", function="analyze"][Instruction::Return]', {
    funcName: 'analyze',
    paramCount: 1,
    postReturn: true
  });
  const retCopy = result5;
  
  let cstate = getOrCreateAsyncState(0);
  cstate.mayLeave = false;
  postReturn0(ret);
  cstate.mayLeave = true;
  return retCopy;
  
}
let exports1Cm32p2AnalyzeFile;

function analyzeFile(arg0) {
  var ptr0 = utf8Encode(arg0, realloc0, memory0);
  var len0 = utf8EncodedLen;
  _debugLog('[iface="analyze-file", function="analyze-file"] [Instruction::CallWasm] (async? false, @ enter)');
  const _wasm_call_currentTaskID = startCurrentTask(0, false, 'exports1Cm32p2AnalyzeFile');
  const ret = exports1Cm32p2AnalyzeFile(ptr0, len0);
  endCurrentTask(0);
  let variant7;
  switch (dataView(memory0).getUint8(ret + 0, true)) {
    case 0: {
      var len5 = dataView(memory0).getUint32(ret + 8, true);
      var base5 = dataView(memory0).getUint32(ret + 4, true);
      var result5 = [];
      for (let i = 0; i < len5; i++) {
        const base = base5 + i * 40;
        var ptr1 = dataView(memory0).getUint32(base + 12, true);
        var len1 = dataView(memory0).getUint32(base + 16, true);
        var result1 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr1, len1));
        var ptr2 = dataView(memory0).getUint32(base + 20, true);
        var len2 = dataView(memory0).getUint32(base + 24, true);
        var result2 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr2, len2));
        let variant4;
        switch (dataView(memory0).getUint8(base + 28, true)) {
          case 0: {
            variant4 = undefined;
            break;
          }
          case 1: {
            var ptr3 = dataView(memory0).getUint32(base + 32, true);
            var len3 = dataView(memory0).getUint32(base + 36, true);
            var result3 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr3, len3));
            variant4 = result3;
            break;
          }
          default: {
            throw new TypeError('invalid variant discriminant for option');
          }
        }
        result5.push({
          line: dataView(memory0).getInt32(base + 0, true) >>> 0,
          col: dataView(memory0).getInt32(base + 4, true) >>> 0,
          severity: clampGuest(dataView(memory0).getUint8(base + 8, true), 0, 255),
          code: result1,
          message: result2,
          suggestion: variant4,
        });
      }
      variant7= {
        tag: 'ok',
        val: result5
      };
      break;
    }
    case 1: {
      var ptr6 = dataView(memory0).getUint32(ret + 4, true);
      var len6 = dataView(memory0).getUint32(ret + 8, true);
      var result6 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr6, len6));
      variant7= {
        tag: 'err',
        val: result6
      };
      break;
    }
    default: {
      throw new TypeError('invalid variant discriminant for expected');
    }
  }
  _debugLog('[iface="analyze-file", function="analyze-file"][Instruction::Return]', {
    funcName: 'analyze-file',
    paramCount: 1,
    postReturn: true
  });
  const retCopy = variant7;
  
  let cstate = getOrCreateAsyncState(0);
  cstate.mayLeave = false;
  postReturn1(ret);
  cstate.mayLeave = true;
  
  
  
  if (typeof retCopy === 'object' && retCopy.tag === 'err') {
    throw new ComponentError(retCopy.val);
  }
  return retCopy.val;
  
}
let exports1Cm32p2DetectSmells;

function detectSmells(arg0) {
  var ptr0 = utf8Encode(arg0, realloc0, memory0);
  var len0 = utf8EncodedLen;
  _debugLog('[iface="detect-smells", function="detect-smells"] [Instruction::CallWasm] (async? false, @ enter)');
  const _wasm_call_currentTaskID = startCurrentTask(0, false, 'exports1Cm32p2DetectSmells');
  const ret = exports1Cm32p2DetectSmells(ptr0, len0);
  endCurrentTask(0);
  var len2 = dataView(memory0).getUint32(ret + 16, true);
  var base2 = dataView(memory0).getUint32(ret + 12, true);
  var result2 = [];
  for (let i = 0; i < len2; i++) {
    const base = base2 + i * 8;
    var ptr1 = dataView(memory0).getUint32(base + 0, true);
    var len1 = dataView(memory0).getUint32(base + 4, true);
    var result1 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr1, len1));
    result2.push(result1);
  }
  _debugLog('[iface="detect-smells", function="detect-smells"][Instruction::Return]', {
    funcName: 'detect-smells',
    paramCount: 1,
    postReturn: true
  });
  const retCopy = {
    complexityScore: dataView(memory0).getInt32(ret + 0, true) >>> 0,
    maintainabilityIndex: dataView(memory0).getInt32(ret + 4, true) >>> 0,
    debtMinutes: dataView(memory0).getInt32(ret + 8, true) >>> 0,
    smellTypes: result2,
  };
  
  let cstate = getOrCreateAsyncState(0);
  cstate.mayLeave = false;
  postReturn2(ret);
  cstate.mayLeave = true;
  return retCopy;
  
}
let exports1Cm32p2PredictIssues;

function predictIssues(arg0, arg1) {
  var ptr0 = utf8Encode(arg0, realloc0, memory0);
  var len0 = utf8EncodedLen;
  var vec2 = arg1;
  var len2 = vec2.length;
  var result2 = realloc0(0, 0, 4, len2 * 8);
  for (let i = 0; i < vec2.length; i++) {
    const e = vec2[i];
    const base = result2 + i * 8;var ptr1 = utf8Encode(e, realloc0, memory0);
    var len1 = utf8EncodedLen;
    dataView(memory0).setUint32(base + 4, len1, true);
    dataView(memory0).setUint32(base + 0, ptr1, true);
  }
  _debugLog('[iface="predict-issues", function="predict-issues"] [Instruction::CallWasm] (async? false, @ enter)');
  const _wasm_call_currentTaskID = startCurrentTask(0, false, 'exports1Cm32p2PredictIssues');
  const ret = exports1Cm32p2PredictIssues(ptr0, len0, result2, len2);
  endCurrentTask(0);
  var len7 = dataView(memory0).getUint32(ret + 4, true);
  var base7 = dataView(memory0).getUint32(ret + 0, true);
  var result7 = [];
  for (let i = 0; i < len7; i++) {
    const base = base7 + i * 40;
    var ptr3 = dataView(memory0).getUint32(base + 12, true);
    var len3 = dataView(memory0).getUint32(base + 16, true);
    var result3 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr3, len3));
    var ptr4 = dataView(memory0).getUint32(base + 20, true);
    var len4 = dataView(memory0).getUint32(base + 24, true);
    var result4 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr4, len4));
    let variant6;
    switch (dataView(memory0).getUint8(base + 28, true)) {
      case 0: {
        variant6 = undefined;
        break;
      }
      case 1: {
        var ptr5 = dataView(memory0).getUint32(base + 32, true);
        var len5 = dataView(memory0).getUint32(base + 36, true);
        var result5 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr5, len5));
        variant6 = result5;
        break;
      }
      default: {
        throw new TypeError('invalid variant discriminant for option');
      }
    }
    result7.push({
      line: dataView(memory0).getInt32(base + 0, true) >>> 0,
      col: dataView(memory0).getInt32(base + 4, true) >>> 0,
      severity: clampGuest(dataView(memory0).getUint8(base + 8, true), 0, 255),
      code: result3,
      message: result4,
      suggestion: variant6,
    });
  }
  _debugLog('[iface="predict-issues", function="predict-issues"][Instruction::Return]', {
    funcName: 'predict-issues',
    paramCount: 1,
    postReturn: true
  });
  const retCopy = result7;
  
  let cstate = getOrCreateAsyncState(0);
  cstate.mayLeave = false;
  postReturn3(ret);
  cstate.mayLeave = true;
  return retCopy;
  
}
let exports1Cm32p2GetPerformanceStats;

function getPerformanceStats() {
  _debugLog('[iface="get-performance-stats", function="get-performance-stats"] [Instruction::CallWasm] (async? false, @ enter)');
  const _wasm_call_currentTaskID = startCurrentTask(0, false, 'exports1Cm32p2GetPerformanceStats');
  const ret = exports1Cm32p2GetPerformanceStats();
  endCurrentTask(0);
  var len1 = dataView(memory0).getUint32(ret + 4, true);
  var base1 = dataView(memory0).getUint32(ret + 0, true);
  var result1 = [];
  for (let i = 0; i < len1; i++) {
    const base = base1 + i * 24;
    var ptr0 = dataView(memory0).getUint32(base + 0, true);
    var len0 = dataView(memory0).getUint32(base + 4, true);
    var result0 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr0, len0));
    result1.push({
      operation: result0,
      durationMs: dataView(memory0).getInt32(base + 8, true) >>> 0,
      memoryUsed: dataView(memory0).getInt32(base + 12, true) >>> 0,
      timestamp: BigInt.asUintN(64, dataView(memory0).getBigInt64(base + 16, true)),
    });
  }
  _debugLog('[iface="get-performance-stats", function="get-performance-stats"][Instruction::Return]', {
    funcName: 'get-performance-stats',
    paramCount: 1,
    postReturn: true
  });
  const retCopy = result1;
  
  let cstate = getOrCreateAsyncState(0);
  cstate.mayLeave = false;
  postReturn4(ret);
  cstate.mayLeave = true;
  return retCopy;
  
}

const $init = (() => {
  let gen = (function* init () {
    const module0 = base64Compile('AGFzbQEAAAABMAhgBX9/f39+AGADf39/AGAFf39/f38AYAJ/fwF/YAF/AGAEf39/fwF/YAABf2AAAALCAQQfY20zMnAyfHZzY29kZTpleGFtcGxlL3RlbGVtZXRyeQtlbWl0LW1ldHJpYwAAHmNtMzJwMnx2c2NvZGU6ZXhhbXBsZS9ob3N0LWFwaQ1nZXQtZmlsZS1pbmZvAAEeY20zMnAyfHZzY29kZTpleGFtcGxlL2hvc3QtYXBpEXJlYWQtZmlsZS1zbmlwcGV0AAIeY20zMnAyfHZzY29kZTpleGFtcGxlL2hvc3QtYXBpD2VtaXQtZGlhZ25vc3RpYwACAw0MAwQDBAMEBQQGBAUHBQMBAAAHwwIND2NtMzJwMnx8YW5hbHl6ZQAEFGNtMzJwMnx8YW5hbHl6ZV9wb3N0AAUUY20zMnAyfHxhbmFseXplLWZpbGUABhljbTMycDJ8fGFuYWx5emUtZmlsZV9wb3N0AAcVY20zMnAyfHxkZXRlY3Qtc21lbGxzAAgaY20zMnAyfHxkZXRlY3Qtc21lbGxzX3Bvc3QACRZjbTMycDJ8fHByZWRpY3QtaXNzdWVzAAobY20zMnAyfHxwcmVkaWN0LWlzc3Vlc19wb3N0AAsdY20zMnAyfHxnZXQtcGVyZm9ybWFuY2Utc3RhdHMADCJjbTMycDJ8fGdldC1wZXJmb3JtYW5jZS1zdGF0c19wb3N0AA0NY20zMnAyX21lbW9yeQIADmNtMzJwMl9yZWFsbG9jAA4RY20zMnAyX2luaXRpYWxpemUADworDAMAAAsCAAsDAAALAgALAwAACwIACwMAAAsCAAsDAAALAgALAwAACwIACwAvCXByb2R1Y2VycwEMcHJvY2Vzc2VkLWJ5AQ13aXQtY29tcG9uZW50BzAuMjM2LjE');
    const module1 = base64Compile('AGFzbQEAAAABHwRgBX9/f39+AGADf39/AGAFf39/f38AYAV/f39/fwADBQQAAQIDBAUBcAEEBAccBQEwAAABMQABATIAAgEzAAMIJGltcG9ydHMBAApFBBEAIAAgASACIAMgBEEAEQAACw0AIAAgASACQQERAQALEQAgACABIAIgAyAEQQIRAgALEQAgACABIAIgAyAEQQMRAwALAC8JcHJvZHVjZXJzAQxwcm9jZXNzZWQtYnkBDXdpdC1jb21wb25lbnQHMC4yMzYuMQD/AQRuYW1lABMSd2l0LWNvbXBvbmVudDpzaGltAeIBBAA0aW5kaXJlY3QtY20zMnAyfHZzY29kZTpleGFtcGxlL3RlbGVtZXRyeS1lbWl0LW1ldHJpYwE1aW5kaXJlY3QtY20zMnAyfHZzY29kZTpleGFtcGxlL2hvc3QtYXBpLWdldC1maWxlLWluZm8COWluZGlyZWN0LWNtMzJwMnx2c2NvZGU6ZXhhbXBsZS9ob3N0LWFwaS1yZWFkLWZpbGUtc25pcHBldAM3aW5kaXJlY3QtY20zMnAyfHZzY29kZTpleGFtcGxlL2hvc3QtYXBpLWVtaXQtZGlhZ25vc3RpYw');
    const module2 = base64Compile('AGFzbQEAAAABHwRgBX9/f39+AGADf39/AGAFf39/f38AYAV/f39/fwACJAUAATAAAAABMQABAAEyAAIAATMAAwAIJGltcG9ydHMBcAEEBAkKAQBBAAsEAAECAwAvCXByb2R1Y2VycwEMcHJvY2Vzc2VkLWJ5AQ13aXQtY29tcG9uZW50BzAuMjM2LjEAHARuYW1lABUUd2l0LWNvbXBvbmVudDpmaXh1cHM');
    const module3 = base64Compile('AGFzbQEAAAABBAFgAAACBQEAAAAACAEA');
    ({ exports: exports0 } = yield instantiateCore(yield module1));
    ({ exports: exports1 } = yield instantiateCore(yield module0, {
      'cm32p2|vscode:example/host-api': {
        'emit-diagnostic': exports0['3'],
        'get-file-info': exports0['1'],
        'read-file-snippet': exports0['2'],
      },
      'cm32p2|vscode:example/telemetry': {
        'emit-metric': exports0['0'],
      },
    }));
    memory0 = exports1.cm32p2_memory;
    realloc0 = exports1.cm32p2_realloc;
    ({ exports: exports2 } = yield instantiateCore(yield module2, {
      '': {
        $imports: exports0.$imports,
        '0': trampoline0,
        '1': trampoline1,
        '2': trampoline2,
        '3': trampoline3,
      },
    }));
    ({ exports: exports3 } = yield instantiateCore(yield module3, {
      '': {
        '': exports1.cm32p2_initialize,
      },
    }));
    postReturn0 = exports1['cm32p2||analyze_post'];
    postReturn1 = exports1['cm32p2||analyze-file_post'];
    postReturn2 = exports1['cm32p2||detect-smells_post'];
    postReturn3 = exports1['cm32p2||predict-issues_post'];
    postReturn4 = exports1['cm32p2||get-performance-stats_post'];
    exports1Cm32p2Analyze = exports1['cm32p2||analyze'];
    exports1Cm32p2AnalyzeFile = exports1['cm32p2||analyze-file'];
    exports1Cm32p2DetectSmells = exports1['cm32p2||detect-smells'];
    exports1Cm32p2PredictIssues = exports1['cm32p2||predict-issues'];
    exports1Cm32p2GetPerformanceStats = exports1['cm32p2||get-performance-stats'];
  })();
  let promise, resolve, reject;
  function runNext (value) {
    try {
      let done;
      do {
        ({ value, done } = gen.next(value));
      } while (!(value instanceof Promise) && !done);
      if (done) {
        if (resolve) resolve(value);
        else return value;
      }
      if (!promise) promise = new Promise((_resolve, _reject) => (resolve = _resolve, reject = _reject));
      value.then(runNext, reject);
    }
    catch (e) {
      if (reject) reject(e);
      else throw e;
    }
  }
  const maybeSyncReturn = runNext(null);
  return promise || maybeSyncReturn;
})();

await $init;

export { analyze, analyzeFile, detectSmells, getPerformanceStats, predictIssues,  }