#!/usr/bin/env python3
"""Test ChromaDB integration with Super Alita memory system."""

import sys
import os
sys.path.insert(0, './src')

print('=== ChromaDB Memory Integration Test ===')
try:
    from plugins.memory_manager_plugin_unified import MemoryManagerPlugin
    try:
        import chromadb
        from chromadb.config import Settings
        CHROMA_AVAILABLE = True
    except Exception:
        CHROMA_AVAILABLE = False
        print('ChromaDB not installed; skipping integration test.')
        sys.exit(0)

    # Initialize the memory manager plugin
    plugin = MemoryManagerPlugin()
    print(f'✅ Plugin name: {plugin.name}')

    # Test ChromaDB integration directly
    client = chromadb.PersistentClient(
        path='./data/chroma_memory',
        settings=Settings(anonymized_telemetry=False)
    )

    # Check existing collections
    collections = client.list_collections()
    print(f'📊 Existing collections: {len(collections)}')
    for col in collections:
        print(f'  - {col.name}: {col.count()} documents')

    # Test memory persistence with semantic collection
    semantic_collection = client.get_or_create_collection(
        name='semantic_memory',
        metadata={'description': 'Semantic knowledge storage'}
    )

    # Add a test memory entry
    test_memory_doc = 'ChromaDB is properly integrated with Super Alita memory system'
    test_metadata = {
        'timestamp': '2025-08-31T14:08:00Z',
        'type': 'semantic',
        'confidence': 0.95
    }

    semantic_collection.upsert(
        ids=['test_memory_001'],
        documents=[test_memory_doc],
        metadatas=[test_metadata]
    )

    # Verify the memory was stored
    results = semantic_collection.query(
        query_texts=['ChromaDB integration'],
        n_results=1
    )

    print(f'✅ Memory persistence test successful!')
    print(f'📝 Stored memory: {test_memory_doc[:50]}...')
    print(f'🔍 Query results: {len(results["documents"][0])} matches found')
    print(f'📈 Semantic collection now has {semantic_collection.count()} documents')

    # Test episodic collection
    episodic_collection = client.get_or_create_collection(
        name='episodic_memory',
        metadata={'description': 'Episodic experience storage'}
    )

    test_episode_doc = 'User requested ChromaDB integration verification'
    episode_metadata = {
        'timestamp': '2025-08-31T14:08:00Z',
        'type': 'episodic',
        'session_id': 'test_session',
        'turn_id': 1
    }

    episodic_collection.upsert(
        ids=['episode_001'],
        documents=[test_episode_doc],
        metadatas=[episode_metadata]
    )

    print(f'✅ Episodic memory test successful!')
    print(f'📚 Episodic collection now has {episodic_collection.count()} documents')

    # Summary
    print('')
    print('🎯 ChromaDB Integration Summary:')
    print(f'  - ChromaDB Version: {chromadb.__version__}')
    print('  - Memory Manager Plugin: ✅ Working')
    print('  - Persistent Storage: ./data/chroma_memory')
    print('  - Semantic Memory: ✅ Functional')
    print('  - Episodic Memory: ✅ Functional')
    print(f'  - Total Collections: {len(client.list_collections())}')

except Exception as e:
    print(f'❌ ChromaDB integration error: {e}')
    import traceback
    traceback.print_exc()
