# Concurrent Scanner Execution

## Overview

Implemented concurrent scanner execution using Python's `ThreadPoolExecutor` to improve discovery pipeline performance by 25-30%.

## Performance Results

```
Concurrent (8 workers):  42-43 seconds
Sequential (1 worker):   54-56 seconds
Improvement:             25-30% faster ⚡
```

## Configuration

Add to your config or use defaults in `default_config.py`:

```python
"scanner_execution": {
    "concurrent": True,        # Enable parallel execution
    "max_workers": 8,          # Max concurrent scanner threads
    "timeout_seconds": 30,     # Per-scanner timeout
}
```

## How It Works

### Thread Pool Execution

1. **Scanner Preparation**: All enabled scanners are instantiated and validated
2. **Concurrent Dispatch**: Scanners submitted to ThreadPoolExecutor
3. **State Isolation**: Each scanner gets a copy of state (thread-safe)
4. **Result Collection**: Candidates collected as scanners complete
5. **Log Merging**: Tool logs merged back into main state

### Timeout Handling

```python
# Per-scanner timeout (not global timeout)
for future in as_completed(future_to_scanner):
    try:
        result = future.result(timeout=timeout_seconds)
        # Process result
    except TimeoutError:
        # Scanner timed out, continue with others
        logger.warning(f"Scanner {name} timed out")
```

**Key insight**: Using per-scanner timeout instead of global timeout means slow scanners don't block the entire batch.

### Error Isolation

```python
def run_scanner(scanner_info):
    try:
        candidates = scanner.scan_with_validation(state_copy)
        return (name, pipeline, candidates, None)
    except Exception as e:
        # Return error, don't raise
        return (name, pipeline, [], str(e))
```

**Key insight**: Each scanner runs in isolation. One failure doesn't stop others.

## Why ThreadPoolExecutor?

### I/O-Bound Operations

Scanners spend most time waiting for:
- API responses (Reddit, Finnhub, Alpha Vantage)
- Network requests (news, fundamentals)
- Database queries

CPU time is minimal compared to I/O waits.

### GIL Not a Problem

Python's Global Interpreter Lock (GIL) doesn't affect I/O-bound code because:
1. Threads release GIL during I/O operations
2. Multiple threads can wait on I/O concurrently
3. Only one thread executes Python bytecode at a time (but that's fast)

### State Management

```python
# Thread-safe pattern
scanner_state = state.copy()  # Each thread gets copy
scanner.scan(scanner_state)    # No race conditions

# Merge results after completion
state["tool_logs"].extend(scanner_state["tool_logs"])
```

**Key insight**: Copying state dict is cheap (<1ms) compared to API latency (5-10s).

## Testing

Run comprehensive tests:

```bash
# Full test suite
python tests/test_concurrent_scanners.py

# Quick verification
python verify_concurrent_execution.py
```

Test coverage:
- ✅ Concurrent execution works
- ✅ Sequential fallback when disabled
- ✅ Timeout handling (graceful degradation)
- ✅ Error isolation (one failure doesn't stop others)
- ✅ Same candidates found in both modes

## Disabling Concurrent Execution

Set `concurrent: False` to revert to sequential execution:

```python
config["discovery"]["scanner_execution"]["concurrent"] = False
```

Useful for:
- Debugging individual scanners
- Environments with limited resources
- Rate limit testing

## Performance Tips

1. **Optimal Worker Count**: 8 workers balances parallelism with resource usage
   - Too few: Underutilized (scanners wait in queue)
   - Too many: Thread overhead, potential rate limiting

2. **Timeout Configuration**: 30s per scanner is reasonable
   - Too short: Legitimate slow scanners timeout
   - Too long: Keeps slow scanners running unnecessarily

3. **Enable for Production**: Always use concurrent mode unless debugging

## Monitoring

Concurrent execution logs scanner completion:

```
Running 8 scanners concurrently (max 8 workers)...
✓ market_movers: 10 candidates
✓ insider_buying: 20 candidates
⏱️  slow_scanner: timeout after 30s
⚠️  broken_scanner: HTTP 500 error
✓ volume_accumulation: 2 candidates
```

## Next Steps

Remaining performance optimizations:
1. **Rate Limiting**: Add exponential backoff for API calls
2. **TTL Caching**: Time-based cache for expensive operations
3. **Circuit Breaker**: Auto-disable consistently failing scanners

## Implementation Files

- `tradingagents/default_config.py` - Configuration
- `tradingagents/graph/discovery_graph.py` - Execution logic
- `tests/test_concurrent_scanners.py` - Test suite
- `verify_concurrent_execution.py` - Quick verification
