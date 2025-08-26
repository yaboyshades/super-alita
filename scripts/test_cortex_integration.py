#!/usr/bin/env python3
"""
Test script to verify VS Code → Cortex bridge integration
Simulates VS Code extension events and validates the flow
"""

import json
import pathlib
import time
from datetime import UTC, datetime


# Simulate the VS Code extension emitting events
def emit_test_events():
    """Emit test events to the telemetry.jsonl file"""
    telemetry_path = pathlib.Path.home() / ".super-alita" / "telemetry.jsonl"
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Test events from VS Code extension
    test_events = [
        {
            "id": "test-copilot-query-001",
            "kind": "COPILOT_QUERY",
            "ts": time.time(),
            "actor": "user",
            "payload": {
                "text": "@alita review this code for architectural compliance",
                "language": "python",
                "file_path": "src/main.py",
                "meta": {
                    "PROMPT_VERSION": "2.0.0",
                    "ARCHITECTURE_HASH": "sha256:a3f4b5c6d7e8",
                    "VERIFICATION_MODE": "ACTIVE",
                    "extension_version": "2.0.0",
                    "vscode_version": "1.103.0"
                }
            },
            "schema_version": "v1"
        },
        {
            "id": "test-guardian-audit-001",
            "kind": "TOOL_RUN",
            "ts": time.time(),
            "actor": "agent",
            "payload": {
                "tool": "super_alita_guardian",
                "args": {
                    "file": "chat_message",
                    "rule": "ArchitecturalCompliance",
                    "version": "2.0.0",
                    "findings_count": 2
                },
                "ok": False,
                "findings": [
                    {"rule": "Plugin Architecture", "message": "Missing PluginInterface inheritance"},
                    {"rule": "Event Bus Patterns", "message": "Direct event creation without source_plugin"}
                ],
                "meta": {
                    "PROMPT_VERSION": "2.0.0",
                    "ARCHITECTURE_HASH": "sha256:a3f4b5c6d7e8",
                    "VERIFICATION_MODE": "ACTIVE"
                }
            },
            "schema_version": "v1"
        },
        {
            "id": "test-ide-interaction-001",
            "kind": "IDE_INTERACTION",
            "ts": time.time(),
            "actor": "user",
            "payload": {
                "action": "document_change",
                "file": "src/core/plugin_interface.py",
                "lines_changed": 5,
                "lines_generated": 12,
                "functions_detected": 1,
                "timestamp": int(time.time() * 1000),
                "meta": {
                    "PROMPT_VERSION": "2.0.0",
                    "extension_version": "2.0.0"
                }
            },
            "schema_version": "v1"
        },
        {
            "id": "test-architectural-compliance-001",
            "kind": "ARCHITECTURAL_AUDIT",
            "ts": time.time(),
            "actor": "agent",
            "payload": {
                "tool": "super_alita_guardian",
                "args": {
                    "file": "workspace_audit",
                    "audit_type": "architectural_compliance"
                },
                "ok": True,
                "compliance_score": 85.7,
                "violations": 3,
                "suggestions": [
                    "Analyzed 15 files",
                    "12 compliant",
                    "3 violations found"
                ],
                "meta": {
                    "PROMPT_VERSION": "2.0.0",
                    "ARCHITECTURE_HASH": "sha256:a3f4b5c6d7e8"
                }
            },
            "schema_version": "v1"
        },
        {
            "id": "test-user-feedback-001",
            "kind": "USER_FEEDBACK",
            "ts": time.time(),
            "actor": "user",
            "payload": {
                "rating": 4,
                "comment": "Good architectural guidance, helped fix plugin interface",
                "context": {
                    "context": "guardian_mode_architectural_review",
                    "timestamp": int(time.time() * 1000)
                },
                "timestamp": int(time.time() * 1000),
                "meta": {
                    "PROMPT_VERSION": "2.0.0"
                }
            },
            "schema_version": "v1"
        }
    ]
    
    print(f"📝 Emitting {len(test_events)} test events to {telemetry_path}")
    
    with open(telemetry_path, 'a', encoding='utf-8') as f:
        for i, event in enumerate(test_events):
            f.write(json.dumps(event) + '\n')
            print(f"  ✅ Event {i+1}: {event['kind']} from {event['actor']}")
            time.sleep(0.5)  # Small delay to simulate real usage
    
    print("\n🎯 All test events emitted! Bridge should process them.")
    return telemetry_path

def analyze_telemetry_file(telemetry_path):
    """Analyze the telemetry file contents"""
    if not telemetry_path.exists():
        print(f"❌ Telemetry file not found: {telemetry_path}")
        return
    
    print(f"\n📊 Analyzing telemetry file: {telemetry_path}")
    
    events = []
    with open(telemetry_path, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                event = json.loads(line.strip())
                events.append(event)
            except json.JSONDecodeError as e:
                print(f"  ⚠️ Line {line_num}: Invalid JSON - {e}")
    
    print(f"📈 Total events in file: {len(events)}")
    
    # Group by kind
    by_kind = {}
    by_actor = {}
    
    for event in events:
        kind = event.get('kind', 'UNKNOWN')
        actor = event.get('actor', 'unknown')
        
        by_kind[kind] = by_kind.get(kind, 0) + 1
        by_actor[actor] = by_actor.get(actor, 0) + 1
    
    print("\n📋 Events by type:")
    for kind, count in sorted(by_kind.items()):
        print(f"  {kind}: {count}")
    
    print("\n👥 Events by actor:")
    for actor, count in sorted(by_actor.items()):
        print(f"  {actor}: {count}")
    
    # Show recent events
    print("\n🕒 Most recent events:")
    recent_events = sorted(events, key=lambda x: x.get('ts', 0))[-3:]
    for event in recent_events:
        ts = datetime.fromtimestamp(event.get('ts', 0), tz=UTC)
        print(f"  {ts.strftime('%H:%M:%S')} - {event.get('kind', 'UNKNOWN')} from {event.get('actor', 'unknown')}")

def main():
    """Main test execution"""
    print("🚀 Testing VS Code → Cortex Bridge Integration")
    print("=" * 50)
    
    # Emit test events
    telemetry_path = emit_test_events()
    
    # Wait a bit for bridge to process
    print("\n⏳ Waiting 3 seconds for bridge to process events...")
    time.sleep(3)
    
    # Analyze results
    analyze_telemetry_file(telemetry_path)
    
    print("\n✅ Test complete! Check the Cortex bridge logs for processing confirmation.")
    print("\n💡 Next steps:")
    print("  1. Test the actual VS Code extension by using @alita in chat")
    print("  2. Make some code changes to trigger document events")
    print("  3. Check that Cortex is learning from the telemetry markers")

if __name__ == "__main__":
    main()