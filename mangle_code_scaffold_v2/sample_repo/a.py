import b


def compute(x):
    # intentionally a bit complex
    total = 0
    for i in range(x):
        if i % 2 == 0:
            total += i
        else:
            if i % 3 == 0:
                total -= i
            else:
                try:
                    total += (i // 2)
                except Exception:
                    total += 0
    return total + b.g(total)

def helper():
    return b.g(1)
