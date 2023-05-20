# operacoes com strings

# concatenacao
var_string1 = "Olá, "
var_string2 = "bom dia."

var_string3 = var_string1 + var_string2
print(var_string3)

# multiplicacao
var_string1 = "rs"
var_string2 = var_string1 * 3
print(var_string2)

# escaping
print("Bom dia,\n" + 'Prezado.')

# tamanho
var_string1 = "Olá"
print(len(var_string1))

# maiusculas, minusculas e capitalizacao
var_string1 = "Olá. Bom dia Senhor."

print(var_string1.lower())
print(var_string1.upper())
print(var_string1.capitalize())

# indexação e substrings
print(var_string1[5])
print(var_string1[-1])
print(var_string1[:3])
print(var_string1[5:])
print(var_string1[5:12])
print(var_string1[:])

# operador in
print("dia" in var_string1)
print("tarde" in var_string1)

# formatacao
produto = "arroz"
peso = 2
valor = 4.55

print("O preço de %d kg de %s é %d reais" % (peso, produto, valor))
print("O preço de %d kg de %s é %f reais" % (peso, produto, valor))
print("O preço de %d kg de %s é %.2f reais" % (peso, produto, valor))

