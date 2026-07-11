import math

def mm2degree(mm):
    degree = mm / 500 * (180 / math.pi)
    print(degree)

def mm2degree2(s):
    s = s.split("±")
    mean = float(s[0]) / 500 * (180 / math.pi)
    std = float(s[1]) / 500 * (180 / math.pi)
    print(f"{mean:.2f} ± {std:.2f}")


if __name__ == "__main__":
    mm2degree2("87.65 ± 35.93")

