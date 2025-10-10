#!/usr/bin/env python3
"""Test full consensus streaming response."""

import time

import requests


def test_full_consensus_stream():
    """Test the complete consensus tool execution via streaming."""
    print("🎯 Testing full consensus streaming...")

    # Start a REUG turn with consensus tool request
    start_payload = {
        "message": "Please use the deepconf_consensus tool with the prompt 'What is 2+2?' Use 2 samples with temperature 0.3.",
        "session_id": "test-full-consensus",
    }

    try:
        print("🚀 Starting REUG turn...")
        response = requests.post(
            "http://127.0.0.1:8080/tools/reug_start_turn",
            json=start_payload,
            timeout=30,
        )

        if response.status_code != 200:
            print(
                f"❌ Failed to start: {response.status_code} - {response.text}"
            )
            return

        result = response.json()
        run_id = result.get("run_id")
        print(f"✅ Started with run_id: {run_id}")

        # Stream the complete response
        print("📡 Streaming full response...")
        all_chunks = []
        finished = False
        max_iterations = 30
        iteration = 0

        while not finished and iteration < max_iterations:
            try:
                stream_response = requests.post(
                    "http://127.0.0.1:8080/tools/reug_stream_next",
                    json={"run_id": run_id},
                    timeout=30,
                )

                if stream_response.status_code != 200:
                    print(f"❌ Stream error: {stream_response.status_code}")
                    print(f"   Response: {stream_response.text}")
                    break

                stream_data = stream_response.json()
                chunks = stream_data.get("chunks", [])
                finished = stream_data.get("finished", False)

                if chunks:
                    print(
                        f"📄 Iteration {iteration}: Received {len(chunks)} chunk(s)"
                    )
                    for i, chunk in enumerate(chunks):
                        chunk_str = str(chunk)
                        print(f"   Chunk {i}: {chunk_str[:100]}...")
                        all_chunks.append(chunk_str)
                else:
                    print(f"📄 Iteration {iteration}: No chunks received")

                if finished:
                    print("✅ Stream finished!")
                    break

                iteration += 1
                time.sleep(1)  # Brief pause between requests

            except requests.exceptions.Timeout:
                print("⏰ Stream timeout")
                break
            except Exception as e:
                print(f"❌ Stream error: {e}")
                break

        print("\n📊 Summary:")
        print(f"   Total chunks: {len(all_chunks)}")
        print(f"   Iterations: {iteration}")
        print(f"   Finished: {finished}")

        # Look for consensus tool calls in the chunks
        consensus_chunks = [
            c
            for c in all_chunks
            if "consensus" in c.lower() or "deepconf" in c.lower()
        ]
        if consensus_chunks:
            print(f"   Consensus-related chunks: {len(consensus_chunks)}")
            for i, chunk in enumerate(consensus_chunks[:3]):  # Show first 3
                print(f"     {i+1}: {chunk[:200]}...")

        return all_chunks

    except Exception as e:
        print(f"❌ Test error: {e}")
        return []


if __name__ == "__main__":
    chunks = test_full_consensus_stream()
    if chunks:
        print(
            f"\n🎉 Successfully received {len(chunks)} chunks from consensus test!"
        )
    else:
        print("\n❌ No chunks received")
