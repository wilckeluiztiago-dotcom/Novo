# 🚀 Simulador de Lançamento — Foguete tipo Saturno V (NASA)

Projeto de portfólio em **Python + Pygame** que simula um lançamento orbital 2D com equações diferenciais e animação em tempo real.

**Destaques:**
- Dinâmica com **massa variável** (foguete que queima propelente)
- **3 estágios** com separação automática (inspirado no Saturno V)
- **Gravidade variável** com altitude
- **Atmosfera exponencial** + força de arrasto
- Integração numérica **Runge–Kutta 4ª ordem (RK4)**
- Animação em tempo real com HUD e rastro

Tudo em um único arquivo Python ✅

---

## 📦 Instalação

Crie um ambiente virtual (opcional, mas recomendado):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Instale dependências:

```bash
pip install numpy pygame pandas
```

---

## ▶️ Como rodar

```bash
python simulador_foguete_saturno_pygame.py
```

**Teclas:**
- `ESPAÇO` → pausa/continua
- `R` → reinicia
- `+ / -` → acelera/desacelera simulação
- `ESC` → sair

---

## 🧠 Modelo matemático (resumo)

A dinâmica 2D é dada por:

**Estado:**  
\[
\mathbf{s}(t) = (x, y, v_x, v_y, m)
\]

**Equações:**
\[
\dot{x} = v_x,\quad \dot{y} = v_y
\]
\[
\dot{v}_x = \frac{F_T \cos\theta - D \frac{v_x}{v}}{m}
\]
\[
\dot{v}_y = \frac{F_T \sin\theta - D \frac{v_y}{v}}{m} - g(y)
\]
\[
\dot{m} = -\dot{m}_p = - \frac{F_T}{I_{sp} g_0}
\]

**Arrasto:**
\[
D = \frac{1}{2}\rho(y) v^2 C_d A
\]
\[
\rho(y) = \rho_0 e^{-y/H}
\]

**Gravidade variável:**
\[
g(y)=\frac{\mu}{(R_T + y)^2}
\]

O empuxo muda ao longo do pitch (gravity turn).

---

## 📊 Saídas

Ao sair do simulador, ele cria:

```
saidas/
 └─ trajetoria.csv
```

Com altitude, velocidade, massa e estágio ao longo do tempo.

---

## 🔥 Por que isso é forte para recrutadores?

- Mostra **EDOs reais de engenharia aeroespacial**
- Integração numérica robusta (RK4)
- Sistema com **staging, arrasto, atmosfera e pitch program**
- Visualização em tempo real (Pygame)
- Código organizado, comentado e pronto para portfólio

---

## 👤 Autor
**Luiz Tiago Wilcke (LT)**  
GitHub: https://github.com/wilckeluiztiago-dotcom/Novo  
E-mail: wilckeluiztiago@gmail.com