from __future__ import division
from sympy import *
from timeit import Timer
import mpmath
import math
import matplotlib.pyplot as plt

x, y, z, t = symbols('x y z t')
COEF_ADAMS_BASH = [
    [1],
    [1],
    [3/2, -1/2],
    [23/12, -4/3, 5/12],
    [55/24, -59/24, 37/24, -3/8],
    [1901/720, -1387/360, 109/30, -637/360, 251/720],
    [4277/1440, -2641/480, 4991/720, -3649/720, 959/480, -95/288],
    [198721/60480, -18637/2520, 235183/20160, -10754/945, 135713/20160, -5603/2520, 19087/60480],
    [16083/4480, -1152169/120960, 242653/13440, -296053/13440, 2102243/120960, -115747/13440, 32863/13440, -5257/17280]
]
COEF_ADAM_MOULTON = [
    [1],
    [1],
    [1/2, 1/2],
    [5/12, 2/3, -1/12],
    [3/8, 19/24, -5/24, 1/24],
    [251/720, 323/360, -11/30, 53/360, -19/720],
    [95/288, 1427/1440, -133/240, 241/720, -173/1440, 3/160],
    [19087/60480, 2713/2520, -15487/20160, 586/945, -6737/20160, 263/2520, -863/60480],
    [5257/17280, 139849/120960, -4511/4480, 123133/120960, -88547/120960, 1537/4480, -11351/120960, 275/24192]
]
COEF_FORM_INV = [
    [1],
    [1, 1],
    [2/3, 4/3, -1/3],
    [6/11, 18/11, -9/11, 2/11],
    [12/25, 48/25, -36/25, 16/25, -3/25],
    [60/137, 300/137, -300/137, 200/137, -75/137, 12/137],
    [60/147, 360/147, -450/147, 400/147, -225/147, 72/147, -10/147]
]

SHOW_GRAPHS = True ## MUDAR PARA TRUE CASO DESEJE VER OS GRAFICOS!
BY_LISTA_VALORES = 0
BY_EULER = 1
BY_EULER_INVERSO = 2
BY_EULER_APRIMORADO = 3
BY_RUNGE_KUTTA = 4

def F(f, t0, y0):
    return f.subs({t : t0, y : y0})

def f_bashforth(funcao, valores_iniciais, passo_atual, h, t0, ordem):
    retorno = valores_iniciais[passo_atual - 1]
    k = int(0)

    for i in range(passo_atual - 1, passo_atual - ordem - 1, -1):
        fn = F(funcao, t0 + i * h, valores_iniciais[i])
        retorno += h * (fn * COEF_ADAMS_BASH[ordem][k])
        k += 1
        
    return retorno

def f_moulton(funcao, valores_iniciais, passo_atual, h, t0, ordem):
    retorno = valores_iniciais[passo_atual - 1]
    k = int(0)

    for i in range(passo_atual, passo_atual - ordem, -1):
        fn = F(funcao, t0 + i * h, valores_iniciais[i])
        retorno += h * (fn * COEF_ADAM_MOULTON[ordem][k])
        k += 1
    
    return retorno

def f_inversa(funcao, valores_iniciais, passo_atual, h, t0, ordem):
    y1 = valores_iniciais[passo_atual]
    t1 = t0 + (passo_atual) * h
    retorno = COEF_FORM_INV[ordem][0] * h * F(funcao, t1, y1)
    k = 1

    for i in range(passo_atual - 1, passo_atual - ordem - 1, -1):
        yn = valores_iniciais[i]
        retorno += (yn * COEF_FORM_INV[ordem][k])
        k += 1
    
    return retorno

def print_header(nome_metodo, y0, t0, h):
    print('Metodo de ' + nome_metodo)
    print('y(' + str(t0) + ') = ' + str(y0))
    print('h = ' + str(h))

def mostra_grafico(nome_metodo, tempos, resultados):
    if (SHOW_GRAPHS == 0):
        return
    
    plt.title('Metodo ' + nome_metodo)
    plt.plot(tempos, resultados)
    plt.show()

def euler_simples(y0, t0, h, qtd_passos, funcao, flag):
    if (flag == 0):
        print_header('Euler Simples', y0, t0, h)

    qtd_passos = int(qtd_passos)
    y0 = float(y0)
    t0 = float(t0)
    h = float(h)
    f = sympify(funcao)

    resultados = []
    tempos = []

    if (flag == 0):
        print('0 ' + str(y0))

    resultados.append(y0)
    tempos.append(t0)

    for passo_atual in range(1, qtd_passos + 1):
        ft0y0 = F(f, t0, y0)
        y1 = y0 + ft0y0 * h
        t1 = t0 + h
        if (flag == 0):
            print(str(passo_atual) + ' ' + str(y1))
        y0 = y1
        t0 = t1
        resultados.append(y0)
        tempos.append(t0)
    
    if (flag == 0):
        mostra_grafico('Euler Simples', tempos, resultados)

    return resultados

def euler_inverso(y0, t0, h, qtd_passos, funcao, flag):
    if (flag == 0):
        print_header('Euler Inverso', y0, t0, h)

    qtd_passos = int(qtd_passos)
    y0 = float(y0)
    t0 = float(t0)
    h = float(h)
    f = sympify(funcao)

    resultados = []
    tempos = []

    if (flag == 0):
        print('0 ' + str(y0))

    resultados.append(y0)
    tempos.append(t0)

    for passo_atual in range(1, qtd_passos + 1):
        y1 = y0 + F(f, t0, y0) * h
        t1 = t0 + h
        y1 = y0 + (h * F(f, t1, y1))
        # eq = y0 + (h * F(f, t0 + h, y)) (resolucao implicita)
        # eq = y - eq
        # y1 = solve(eq, y).pop()
        if (flag == 0):
            print(str(passo_atual) + ' ' + str(y1))
        y0 = y1
        t0 = t1
        resultados.append(y0)
        tempos.append(t0)
    
    if (flag == 0):
        mostra_grafico('Euler Inverso', tempos, resultados)

    return resultados

def euler_aprimorado(y0, t0, h, qtd_passos, funcao, flag):
    if (flag == 0):
        print_header('Euler Aprimorado', y0, t0, h)

    resultados = []
    tempos = []

    qtd_passos = int(qtd_passos)
    y0 = float(y0)
    t0 = float(t0)
    h = float(h)
    f = sympify(funcao)

    if (flag == 0):
        print('0 ' + str(y0))

    resultados.append(y0)
    tempos.append(t0)

    for passo_atual in range(1, qtd_passos + 1):
        ft0y0 = F(f, t0, y0)
        y1 = y0 + ft0y0 * h
        t1 = t0 + h
        ft1y1 = F(f, t1, y1)

        y1 = y0 + ((ft0y0 + ft1y1) * (h)) / 2.0

        if (flag == 0):
            print(str(passo_atual) + ' ' + str(y1))

        y0 = y1
        t0 = t1
        resultados.append(y0)
        tempos.append(t0)
    
    if (flag == 0):
        mostra_grafico('Euler Aprimorado', tempos, resultados)

    return resultados

def runge_kutta(y0, t0, h, qtd_passos, funcao, flag):
    if (flag == 0):
        print_header('Runge-Kutta', y0, t0, h)
    
    resultados = []
    tempos = []

    qtd_passos = int(qtd_passos)
    y0 = float(y0)
    t0 = float(t0)
    h = float(h)
    f = sympify(funcao)

    if (flag == 0):
        print('0 ' + str(y0))

    resultados.append(y0)
    tempos.append(t0)

    for passo_atual in range (1, qtd_passos + 1):
        k1 = F(f, t0, y0)
        k2 = F(f, t0 + h / 2.0, y0 + (h * k1) / 2.0)
        k3 = F(f, t0 + h / 2.0, y0 + (h * k2) / 2.0)
        k4 = F(f, t0 + h, y0 + h * k3)

        y1 = y0 + (h / 6.0) * (k1 + 2 * k2 + 2.0 * k3 + k4)
        
        if (flag == 0):
            print(str(passo_atual) + ' ' + str(y1))

        y0 = y1
        t0 = t0 + h
        resultados.append(y0)
        tempos.append(t0)

    if (flag == 0):
        mostra_grafico('Runge-Kutta', tempos, resultados)

    return resultados

def adams_bashforth(entrada, t0, h, qtd_passos, funcao, ordem, origem):
    resultados = []
    tempos = []

    qtd_passos = int(qtd_passos)
    t0 = float(t0)
    h = float(h)
    f = sympify(funcao)
    ordem = int(ordem)
    origem = int(origem)

    valores_iniciais = []
    nome_metodo = ""

    if (origem == BY_LISTA_VALORES):
        valores_iniciais = entrada
        nome_metodo = 'Adams-Bashforth'
    elif (origem == BY_EULER):
        valores_iniciais = euler_simples(entrada, t0, h, ordem - 1, funcao, 1)
        nome_metodo = 'Adams-Bashforth por Euler Simples'
    elif (origem == BY_EULER_INVERSO):
        valores_iniciais = euler_inverso(entrada, t0, h, ordem - 1, funcao, 1)
        nome_metodo = 'Adams-Bashforth por Euler Inverso'
    elif (origem == BY_EULER_APRIMORADO):
        valores_iniciais = euler_aprimorado(entrada, t0, h, ordem - 1, funcao, 1)
        nome_metodo = 'Adams-Bashforth por Euler Aprimorado'
    elif (origem == BY_RUNGE_KUTTA):
        valores_iniciais = runge_kutta(entrada, t0, h, ordem - 1, funcao, 1)
        nome_metodo = 'Adams-Bashforth por Runge-Kutta (ordem = ' + str(ordem) + ')'

    print_header(nome_metodo, valores_iniciais[0], t0, h)

    for i in range(ordem):
        valores_iniciais[i] = float(valores_iniciais[i])
        print(str(i) + ' ' + str(valores_iniciais[i]))
        resultados.append(valores_iniciais[i])
        tempos.append(float(t0 + i * h))
    
    for passo_atual in range(ordem, qtd_passos + 1):
        y1 = f_bashforth(f, valores_iniciais, passo_atual, h, t0, ordem)
        print(str(passo_atual) + ' ' + str(y1))
        valores_iniciais.append(y1)
        resultados.append(y1)
        tempos.append(t0 + passo_atual * h)
    
    mostra_grafico(nome_metodo, tempos, resultados)

    return resultados

def adam_moulton(y0, t0, h, qtd_passos, funcao, ordem, origem):    
    resultados = []
    tempos = []

    qtd_passos = int(qtd_passos)
    t0 = float(t0)
    h = float(h)
    f = sympify(funcao)
    ordem = int(ordem)

    valores_iniciais = []
    nome_metodo = ""

    if (origem == BY_LISTA_VALORES):
        valores_iniciais = y0
        nome_metodo = 'Adam-Moulton'
    elif (origem == BY_EULER):
        valores_iniciais = euler_simples(y0, t0, h, ordem - 2, funcao, 1)
        nome_metodo = 'Adam-Moulton por Euler Simples'
    elif (origem == BY_EULER_INVERSO):
        valores_iniciais = euler_inverso(y0, t0, h, ordem - 2, funcao, 1)
        nome_metodo = 'Adam-Moulton por Euler Inverso'
    elif (origem == BY_EULER_APRIMORADO):
        valores_iniciais = euler_aprimorado(y0, t0, h, ordem - 2, funcao, 1)
        nome_metodo = 'Adam-Moulton por Euler Aprimorado'
    elif (origem == BY_RUNGE_KUTTA):
        valores_iniciais = runge_kutta(y0, t0, h, ordem - 2, funcao, 1)
        nome_metodo = 'Adam-Moulton por Runge-Kutta (ordem = ' + str(ordem) + ')'

    print_header(nome_metodo, valores_iniciais[0], t0, h)

    for i in range(ordem - 1):
        valores_iniciais[i] = float(valores_iniciais[i])
        print(str(i) + ' ' + str(valores_iniciais[i]))
        resultados.append(valores_iniciais[i])
        tempos.append(t0 + i * h)
    
    for passo_atual in range(ordem - 1, qtd_passos + 1):
        y1 = f_bashforth(f, valores_iniciais, passo_atual, h, t0, ordem - 1) #previsao a partir de a. bash.
        valores_iniciais.append(y1)
        y1 = f_moulton(f, valores_iniciais, passo_atual, h, t0, ordem)
        valores_iniciais.pop()
        print(str(passo_atual) + ' ' + str(y1))
        valores_iniciais.append(float(y1))
        resultados.append(y1)
        tempos.append(t0 + passo_atual * h)

    mostra_grafico(nome_metodo, tempos, resultados)
    
    return resultados

def formula_inversa(y0, t0, h, qtd_passos, funcao, ordem, origem):    
    resultados = []
    tempos = []

    qtd_passos = int(qtd_passos)
    t0 = float(t0)
    h = float(h)
    f = sympify(funcao)
    ordem = int(ordem)

    valores_iniciais = []
    nome_metodo = ""

    if (origem == BY_LISTA_VALORES):
        valores_iniciais = y0
        nome_metodo = 'Formula Inversa de Dif.'
    elif (origem == BY_EULER):
        valores_iniciais = euler_simples(y0, t0, h, ordem - 2, funcao, 1)
        nome_metodo = 'Formula Inversa de Dif. por Euler Simples'
    elif (origem == BY_EULER_INVERSO):
        valores_iniciais = euler_inverso(y0, t0, h, ordem - 2, funcao, 1)
        nome_metodo = 'Formula Inversa de Dif. por Euler Inverso'
    elif (origem == BY_EULER_APRIMORADO):
        valores_iniciais = euler_aprimorado(y0, t0, h, ordem - 2, funcao, 1)
        nome_metodo = 'Formula Inversa de Dif. por Euler Aprimorado'
    elif (origem == BY_RUNGE_KUTTA):
        valores_iniciais = runge_kutta(y0, t0, h, ordem - 2, funcao, 1)
        nome_metodo = 'Formula Inversa de Dif. por Runge-Kutta (ordem = ' + str(ordem) + ')'

    print_header(nome_metodo, valores_iniciais[0], t0, h)

    for i in range(ordem - 1):
        valores_iniciais[i] = float(valores_iniciais[i])
        print(str(i) + ' ' + str(valores_iniciais[i]))
        resultados.append(valores_iniciais[i])
        tempos.append(t0 + i * h)
    
    for passo_atual in range(ordem - 1, qtd_passos + 1):
        y1 = f_bashforth(f, valores_iniciais, passo_atual, h, t0, ordem - 1) #previsao a partir de a. bash.
        valores_iniciais.append(y1)
        y1 = f_inversa(f, valores_iniciais, passo_atual, h, t0, ordem - 1)
        valores_iniciais.pop()
        print(str(passo_atual) + ' ' + str(y1))
        valores_iniciais.append(float(y1))
        resultados.append(y1)
        tempos.append(t0 + passo_atual * h)

    mostra_grafico(nome_metodo, tempos, resultados)
    
    return resultados

def filtra_entrada(data):
    metodo = data[0]
    euler = metodo.find('euler') != -1
    rk = metodo.find('runge_kutta') != -1
    bashforth = metodo.find('bashforth') != -1
    multon = metodo.find('multon') != -1
    by = metodo.find('by') != -1
    inversa = metodo.find('formula') != -1

    valores_iniciais = []
    ordem = -1

    if (not by and not euler and not rk):
        if (multon or inversa):
            ordem = int(data[len(data) - 1])
            valores_iniciais = []
            for valor in range(1, ordem):
                valores_iniciais.append(data[valor])
        else:
            ordem = int(data[len(data) - 1])
            valores_iniciais = []
            for valor in range(1, ordem + 1):
                valores_iniciais.append(data[valor])

    if (bashforth):
        if (not by):
            return (valores_iniciais, data[ordem + 1], data[ordem + 2], data[ordem + 3], data[ordem + 4], ordem)
        else:
            return (data[1], data[2], data[3], data[4], data[5], data[6])
    elif (multon):
        if (not by):
            return (valores_iniciais, data[ordem], data[ordem + 1], data[ordem + 2], data[ordem + 3], ordem)
        else:
            return (data[1], data[2], data[3], data[4], data[5], data[6])
    elif (inversa):
        if (not by):
            return (valores_iniciais, data[ordem], data[ordem + 1], data[ordem + 2], data[ordem + 3], ordem)
        else:
            return (data[1], data[2], data[3], data[4], data[5], data[6])
    elif (euler):
        return (data[1], data[2], data[3], data[4], data[5], -1)
    elif (rk):
        return (data[1], data[2], data[3], data[4], data[5], -1)
    else:
        print('Metodo nao existente.')

    return (-1, -1, -1, -1, -1, -1)

def calcula_origem(nome_metodo):
    euler_inv = nome_metodo.find('by_euler_inverso')
    euler_apr = nome_metodo.find('by_euler_aprimorado')
    euler_simp = nome_metodo.find('by_euler')
    rk = nome_metodo.find('by_runge_kutta')

    if (euler_inv != -1):
        return BY_EULER_INVERSO
    elif (euler_apr != -1):
        return BY_EULER_APRIMORADO
    elif (euler_simp != -1):
        return BY_EULER
    elif (rk != -1):
        return BY_RUNGE_KUTTA
    else:
        return BY_LISTA_VALORES
    
def main():
    input_file = open('entrada.txt', 'r')
    
    tempos_metodos = []

    for line in input_file:
        data = line.split(' ')

        y0, t0, h, qtd_passos, funcao, ordem = filtra_entrada(data)

        metodo = data[0]
        tempo = 0

        if (metodo == "euler"):
            t = Timer(lambda : euler_simples(y0, t0, h, qtd_passos, funcao, 0))
            tempo = t.timeit(number=1)
        elif (metodo == "euler_inverso"):
            t = Timer(lambda : euler_inverso(y0, t0, h, qtd_passos, funcao, 0))
            tempo = t.timeit(number=1)
        elif (metodo == "euler_aprimorado"):
            t = Timer(lambda : euler_aprimorado(y0, t0, h, qtd_passos, funcao, 0))
            tempo = t.timeit(number=1)
        elif (metodo == "runge_kutta"):
            t = Timer(lambda : runge_kutta(y0, t0, h, qtd_passos, funcao, 0))
            tempo = t.timeit(number=1)
        elif (metodo.find('bashforth') != -1):
            origem = calcula_origem(metodo)
            t = Timer(lambda : adams_bashforth(y0, t0, h, qtd_passos, funcao, ordem, origem))
            tempo = t.timeit(number=1)
        elif (metodo.find('multon') != -1):
            origem = calcula_origem(metodo)
            t = Timer(lambda : adam_moulton(y0, t0, h, qtd_passos, funcao, ordem, origem))
            tempo = t.timeit(number=1)
        else:
            origem = calcula_origem(metodo)
            t = Timer(lambda : formula_inversa(y0, t0, h, qtd_passos, funcao, ordem, origem))
            tempo = t.timeit(number=1)

        print('')
        tempos_metodos.append((metodo, tempo))

    # DESCOMENTAR P/ MOSTRAR O TEMPO DE EXECUÇÃO DE CADA MÉTODO
    # for tempo in tempos_metodos:
    #     print(tempo)
    #     print('---')

if __name__ == "__main__":
    main()