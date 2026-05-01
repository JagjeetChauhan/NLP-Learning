# How to convert characters to numbers?
# ord() returns the Unicode code point.
print(ord("A"))
print(ord("🚀"))
print(ord("क"))

# chr() returns the character from numbers.
print(chr(65))
print(chr(128640))

s = "Hello 😊"
print(len(s))
print(len(s.encode("UTF-8")))

for char in s:
    decimal = (ord(char))
    binary = format(decimal, 'b')
    hexa = format(decimal, 'x')
    print(f"{char} -> Decimal: {decimal} | Binary: {binary} | Hexa: {hexa}")
out_unicode = s.encode("UTF-8")
print(out_unicode)
print([format(byte, '08b') for byte in out_unicode])