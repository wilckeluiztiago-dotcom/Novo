// go_mcts.cpp — Go 19x19 com IA MCTS assíncrona e janela redimensionável
// Autor: Luiz Tiago Wilcke (LT)

#include <SFML/Graphics.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <chrono>

using namespace std;

// =================== Parâmetros de jogo ===================
static constexpr int   TAM   = 19;         // tamanho do tabuleiro
static constexpr double KOMI = 6.5;
static constexpr int   SEMENTE = 42;

// (IA)
static constexpr int   PROF_PLAYOUT_MAX = TAM*TAM + 3*TAM;
static constexpr int   SIMS_MIN         = 64;          // mínimo por lance (garante alguma qualidade)
static constexpr int   TEMPO_MS_PADRAO  = 250;         // orçamento de tempo por lance (IA mais ágil)

// ========== Util aleatório ==========
static std::mt19937& rng_global() {
    static thread_local std::mt19937 rng(SEMENTE);
    return rng;
}

enum Cor { VAZIO=0, PRETO=1, BRANCO=2 };

struct Jogada {
    int x, y;                  // (-1,-1) = passe
    bool passe() const { return x < 0 || y < 0; }
};

// =================== Estado do jogo ===================
struct Estado {
    array<int, TAM*TAM> cel{}; // 0 vazio, 1 preto, 2 branco
    Cor vez = BRANCO;          // HUMANO (brancas) começa
    int captP = 0, captB = 0;
    int posKo = -1;            // índice linear de ko; -1 = sem ko
    int passesSeguidos = 0;

    inline bool dentro(int x,int y) const { return x>=0 && x<TAM && y>=0 && y<TAM; }
    inline int  idx(int x,int y)   const { return y*TAM + x; }
    inline int  get(int x,int y)   const { return cel[idx(x,y)]; }
    inline void set(int x,int y,int v){ cel[idx(x,y)] = v; }

    vector<pair<int,int>> viz(int x,int y) const {
        vector<pair<int,int>> v;
        if (y>0)        v.push_back({x,y-1});
        if (y<TAM-1)    v.push_back({x,y+1});
        if (x>0)        v.push_back({x-1,y});
        if (x<TAM-1)    v.push_back({x+1,y});
        return v;
    }

    // BFS de grupo e liberdades
    void grupoLibs(int sx,int sy, vector<int>& grupo, int& libs) const {
        grupo.clear(); libs=0;
        int cor = get(sx,sy);
        if (cor==VAZIO) return;
        vector<uint8_t> vis(TAM*TAM,0);
        queue<pair<int,int>> q;
        q.push({sx,sy}); vis[idx(sx,sy)]=1;
        while(!q.empty()){
            auto [x,y]=q.front(); q.pop();
            grupo.push_back(idx(x,y));
            for(auto [nx,ny]: viz(x,y)){
                int v = get(nx,ny);
                if(v==VAZIO) libs++;
                else if(v==cor && !vis[idx(nx,ny)]){
                    vis[idx(nx,ny)]=1; q.push({nx,ny});
                }
            }
        }
    }

    int removerGrupo(const vector<int>& grupo){
        int n=0;
        for(int p: grupo){ if(cel[p]!=VAZIO){ cel[p]=VAZIO; n++; } }
        return n;
    }

    bool autoAtariSemCapturar(Cor c,int x,int y) const {
        if(!dentro(x,y) || get(x,y)!=VAZIO) return true;
        Estado t = *this;
        t.set(x,y,c);
        int capt=0;
        for(auto [nx,ny]: viz(x,y)){
            int v = t.get(nx,ny);
            if(v!=VAZIO && v!=c){
                vector<int> g; int libs=0; t.grupoLibs(nx,ny,g,libs);
                if(libs==0) capt += (int)g.size();
            }
        }
        vector<int> g; int libs=0; t.grupoLibs(x,y,g,libs);
        return (libs==1 && capt==0);
    }

    bool movimentoLegal(Cor c, int x,int y) const {
        if(!dentro(x,y)) return false;
        if(get(x,y)!=VAZIO) return false;
        if(posKo == idx(x,y)) return false;
        if(autoAtariSemCapturar(c,x,y)) return false;

        Estado tmp = *this; tmp.posKo = -1; tmp.set(x,y,c);
        int capt=0, ultima=-1;

        for(auto [nx,ny]: viz(x,y)){
            int v = tmp.get(nx,ny);
            if(v!=VAZIO && v!=c){
                vector<int> g; int libs=0; tmp.grupoLibs(nx,ny,g,libs);
                if(libs==0){ capt += tmp.removerGrupo(g); if((int)g.size()==1) ultima=g[0]; }
            }
        }
        vector<int> g; int libs=0; tmp.grupoLibs(x,y,g,libs);
        if(libs==0 && capt==0) return false; // suicídio
        return true;
    }

    void aplicarJogada(const Jogada& j){
        if(j.passe()){
            passesSeguidos++;
            posKo=-1;
            vez = (vez==PRETO?BRANCO:PRETO);
            return;
        }
        int x=j.x,y=j.y;
        set(x,y,vez);
        int capt=0, ultima=-1;

        for(auto [nx,ny]: viz(x,y)){
            int v = get(nx,ny);
            if(v!=VAZIO && v!=vez){
                vector<int> g; int libs=0;
                grupoLibs(nx,ny,g,libs);
                if(libs==0){
                    capt += removerGrupo(g);
                    if((int)g.size()==1) ultima=g[0];
                }
            }
        }

        vector<int> g; int libs=0;
        grupoLibs(x,y,g,libs);
        if(libs==0 && capt==0){ set(x,y,VAZIO); return; }

        if(vez==PRETO) captP += capt; else captB += capt;
        posKo = (capt==1 ? ultima : -1);

        passesSeguidos=0;
        vez = (vez==PRETO?BRANCO:PRETO);
    }

    vector<Jogada> gerarLegais() const {
        vector<Jogada> mv; mv.reserve(TAM*TAM+1);
        for(int y=0;y<TAM;y++) for(int x=0;x<TAM;x++)
            if(movimentoLegal(vez,x,y)) mv.push_back({x,y});
        mv.push_back({-1,-1});
        return mv;
    }

    bool ehOlhoProprio(Cor c,int x,int y) const {
        if(get(x,y)!=VAZIO) return false;
        int same=0, vaz=0;
        for(auto [nx,ny]: viz(x,y)){
            int v=get(nx,ny);
            if(v==c) same++; else if(v==VAZIO) vaz++;
        }
        return (same>=3 && vaz==0);
    }

    static pair<double,double> pontuarChines(const Estado& s){
        double p=s.captP, b=s.captB+KOMI;
        for(int i=0;i<TAM*TAM;i++){
            if(s.cel[i]==PRETO) p++;
            else if(s.cel[i]==BRANCO) b++;
        }
        vector<uint8_t> vis(TAM*TAM,0);
        for(int y=0;y<TAM;y++) for(int x=0;x<TAM;x++){
            if(s.get(x,y)!=VAZIO || vis[s.idx(x,y)]) continue;
            queue<pair<int,int>> q; q.push({x,y});
            vis[s.idx(x,y)]=1;
            int tamReg=0; uint8_t borda=0;
            while(!q.empty()){
                auto [cx,cy]=q.front(); q.pop(); tamReg++;
                for(auto [nx,ny]: s.viz(cx,cy)){
                    int v=s.get(nx,ny);
                    if(v==VAZIO){
                        int id=s.idx(nx,ny);
                        if(!vis[id]){vis[id]=1; q.push({nx,ny});}
                    }else if(v==PRETO) borda|=1;
                    else if(v==BRANCO) borda|=2;
                }
            }
            if(borda==1) p += tamReg;
            else if(borda==2) b += tamReg;
        }
        return {p,b};
    }

    // playout enviesado simples
    int playout(std::mt19937& rng) const {
        Estado s=*this;
        std::uniform_real_distribution<double> U(0.0,1.0);
        int passos=0;
        while(s.passesSeguidos<2 && passos<PROF_PLAYOUT_MAX){
            auto mv = s.gerarLegais();
            vector<Jogada> bons; bons.reserve(32);
            for(auto &j: mv){
                if(j.passe()) continue;
                if(s.ehOlhoProprio(s.vez,j.x,j.y)) continue;
                Estado t=s; t.aplicarJogada(j);
                if(s.vez==PRETO && t.captP > s.captP) { bons.push_back(j); continue; }
                if(s.vez==BRANCO && t.captB > s.captB){ bons.push_back(j); continue; }
                // atari
                bool atari=false;
                for(auto [nx,ny]: s.viz(j.x,j.y)){
                    int v=s.get(nx,ny);
                    if(v!=VAZIO && v!=(int)s.vez){
                        vector<int> g; int libs=0; t.grupoLibs(nx,ny,g,libs);
                        if(libs==1){ atari=true; break; }
                    }
                }
                if(atari) bons.push_back(j);
            }
            Jogada esc;
            if(!bons.empty() && U(rng)<0.85){
                std::uniform_int_distribution<int> D(0,(int)bons.size()-1);
                esc = bons[D(rng)];
            }else{
                std::uniform_int_distribution<int> D(0,(int)mv.size()-1);
                esc = mv[D(rng)];
            }
            s.aplicarJogada(esc);
            passos++;
        }
        auto sc = pontuarChines(s);
        if(sc.first>sc.second) return +1;
        if(sc.first<sc.second) return -1;
        return 0;
    }
};

// =================== MCTS (UCT simples + prior leve) ===================
struct No {
    Jogada jogada{-1,-1};
    double vit=0.0;  // vitórias das PRETAS somadas (+0.5 empate)
    int vis=0;
    vector<unique_ptr<No>> filhos;
    bool exp=false;
    No* pai=nullptr;
};

static double prior(const Estado& s, const Jogada& j){
    if(j.passe()) return 0.01;
    double cx=(TAM-1)/2.0, cy=(TAM-1)/2.0;
    double dx=j.x-cx, dy=j.y-cy;
    double d=hypot(dx,dy);
    double p = 0.3*exp(-0.25*d); // favorecer centro
    static const vector<pair<int,int>> hoshi={{3,3},{9,3},{15,3},{3,9},{9,9},{15,9},{3,15},{9,15},{15,15}};
    for(auto [hx,hy]: hoshi) if(hx==j.x && hy==j.y){ p+=0.25; break; }
    int contato=0; for(auto [nx,ny]: s.viz(j.x,j.y)){ int v=s.get(nx,ny); if(v==PRETO||v==BRANCO) contato++; }
    p += 0.05*contato;
    if(j.x==0||j.y==0||j.x==TAM-1||j.y==TAM-1) p*=0.85; // borda um pouco pior
    return max(0.001, min(0.9, p));
}

struct MCTS {
    double c_ucb=0.9, c_prior=1.6;

    Jogada pensarTempo(const Estado& raizEst, int tempo_ms){
        unique_ptr<No> raiz = make_unique<No>();
        raiz->vis=1; // evita log(0)

        auto inicio = std::chrono::high_resolution_clock::now();
        int sims=0;

        do {
            // Seleção
            vector<No*> cam; cam.push_back(raiz.get());
            Estado s = raizEst;
            No* no = raiz.get();
            while(no->exp && !no->filhos.empty()){
                no = selecionar(no);
                cam.push_back(no);
                if(!no->jogada.passe()){
                    if(!s.movimentoLegal(s.vez, no->jogada.x,no->jogada.y)) break;
                }
                s.aplicarJogada(no->jogada);
            }
            // Expansão
            if(!no->exp){
                No* novo = expandir(no, s);
                if(novo!=no){
                    if(!novo->jogada.passe()) s.aplicarJogada(novo->jogada);
                    no = novo; cam.push_back(no);
                }
            }
            // Simulação
            int r = s.playout(rng_global()); // +1 pretas, -1 brancas
            // Retropropagação
            for(No* p: cam){
                p->vis += 1;
                if(r>0) p->vit += 1.0;
                else if(r==0) p->vit += 0.5;
            }
            sims++;
        } while(
            sims < SIMS_MIN ||
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - inicio
            ).count() < tempo_ms
        );

        // escolhe pelo número de visitas
        No* best=nullptr; int bestv=-1;
        for(auto& f: raiz->filhos) if(f->vis>bestv){ bestv=f->vis; best=f.get(); }
        return best?best->jogada:Jogada{-1,-1};
    }

    No* selecionar(No* no){
        No* esc=nullptr; double best=-1e18;
        double lnN = log((double)max(1,no->vis));
        for(auto& f: no->filhos){
            double Q = (f->vis>0 ? f->vit / f->vis : 0.5);
            double U = c_ucb*sqrt(lnN / (double)max(1,f->vis));
            double P = c_prior * 0.1 * (sqrt((double)no->vis)/(1.0+f->vis)); // reforço leve
            double sc = Q + U + P;
            if(sc>best){ best=sc; esc=f.get(); }
        }
        return esc?esc:no;
    }

    No* expandir(No* no, Estado& s){
        auto todos = s.gerarLegais();
        // marcar já existentes
        vector<pair<int,int>> ja; ja.reserve(no->filhos.size());
        for(auto& f: no->filhos) ja.push_back({f->jogada.x,f->jogada.y});
        vector<Jogada> cand; cand.reserve(todos.size());
        for(auto &m: todos){
            if(find(ja.begin(),ja.end(),pair<int,int>{m.x,m.y})==ja.end())
                cand.push_back(m);
        }
        if(cand.empty()){ no->exp=true; return no; }

        // roleta pelo prior
        vector<double> w; w.reserve(cand.size());
        double soma=0.0;
        for(auto &j: cand){ double pr=prior(s,j); w.push_back(pr); soma+=pr; }
        std::uniform_real_distribution<double> U(0.0,soma);
        double r=U(rng_global()), a=0.0; int k=0;
        for(; k<(int)cand.size(); ++k){ a+=w[k]; if(a>=r) break; }
        if(k==(int)cand.size()) k=(int)cand.size()-1;

        auto filho = make_unique<No>();
        filho->jogada = cand[k];
        filho->pai = no;
        No* ret = filho.get();
        no->filhos.push_back(std::move(filho));
        if((int)no->filhos.size()==(int)todos.size()) no->exp=true;
        return ret;
    }
};

// =================== UI / Layout dinâmico ===================
// Calcula o layout (tudo responde ao redimensionamento)
struct Layout {
    int W, H;           // tamanho da janela
    int ALT_UI = 110;   // altura da barra inferior
    int baseX, baseY;   // canto superior esquerdo da área do tabuleiro
    int board;          // lado do quadrado do tabuleiro (px)
    int margem;         // margem interna
    float cel;          // passo entre interseções
    float raio;         // raio da pedra
    float tol_click;    // tolerância de clique

    explicit Layout(sf::Vector2u sz){
        W = (int)sz.x; H = (int)sz.y;
        // área disponível para o tabuleiro
        int availH = max(300, H - ALT_UI);
        board = min(W, availH);
        // centraliza o tabuleiro
        baseX = (W - board)/2;
        baseY = (availH - board)/2;
        margem = max(24, board/20);       // ~5% do lado
        cel = float(board - 2*margem) / float(TAM - 1);
        raio = cel * 0.42f;
        tol_click = 0.48f*cel;
    }

    inline sf::Vector2f posPixel(int x,int y) const {
        return sf::Vector2f(float(baseX + margem) + x*cel, float(baseY + margem) + y*cel);
    }
    inline bool dentroTab(float px, float py) const {
        return (px >= baseX && px <= baseX+board && py >= baseY && py <= baseY+board);
    }
    inline pair<int,int> posTab(sf::Vector2f p) const {
        float relx = (p.x - (baseX + margem)) / cel;
        float rely = (p.y - (baseY + margem)) / cel;
        int x = (int)std::round(relx);
        int y = (int)std::round(rely);
        x = std::max(0, std::min(TAM-1, x));
        y = std::max(0, std::min(TAM-1, y));
        return {x,y};
    }
};

int main(){
    srand(SEMENTE);

    Estado estado;           // HUMANO (brancas) começa
    MCTS mcts;
    int  tempo_ms_ia = TEMPO_MS_PADRAO;   // pode ajustar em runtime (teclas +/-)

    // Janela redimensionável (Default = permite minimizar/maximizar)
    sf::RenderWindow win(sf::VideoMode(1000, 1000), "Go 19x19 — Pretas (IA) vs Brancas (Você)", sf::Style::Default);
    win.setFramerateLimit(60);

    sf::Font fonte; bool temFonte = fonte.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");

    // Botões (posicionados no desenho, não fixos)
    sf::RectangleShape btPass({120.f,36.f});
    sf::RectangleShape btSug ({120.f,36.f});

    auto drawText=[&](string s,float x,float y,unsigned sz=18,sf::Color c=sf::Color::Black){
        if(!temFonte) return;
        sf::Text t; t.setFont(fonte); t.setString(s); t.setCharacterSize(sz); t.setFillColor(c); t.setPosition(x,y); win.draw(t);
    };

    // IA assíncrona
    bool iaCalculando=false;
    std::future<Jogada> fut;

    auto startIA = [&](){
        iaCalculando=true;
        fut = std::async(std::launch::async, [&](){
            return mcts.pensarTempo(estado, tempo_ms_ia);
        });
    };

    Jogada sugestao{-1,-1};
    bool mostrarSug=false;
    bool jogoEncerrado=false;

    while(win.isOpen()){
        Layout L(win.getSize()); // recalc layout a cada frame (responde a resize)

        // Reposiciona botões na barra inferior
        btPass.setPosition(float(L.W - 140), float(L.H - L.ALT_UI + 20));
        btPass.setSize({120.f,36.f}); btPass.setFillColor(sf::Color(230,230,230));
        btPass.setOutlineThickness(2.f); btPass.setOutlineColor(sf::Color::Black);

        btSug.setPosition (float(L.W - 280), float(L.H - L.ALT_UI + 20));
        btSug.setSize({120.f,36.f}); btSug.setFillColor(sf::Color(200,230,255));
        btSug.setOutlineThickness(2.f); btSug.setOutlineColor(sf::Color::Black);

        // ============ Eventos ============
        sf::Event ev;
        while(win.pollEvent(ev)){
            if(ev.type==sf::Event::Closed) win.close();
            if(ev.type==sf::Event::KeyPressed && ev.key.code==sf::Keyboard::Escape) win.close();

            if(ev.type==sf::Event::KeyPressed){
                // Ajuste de força em tempo real
                if(ev.key.code==sf::Keyboard::Add || ev.key.code==sf::Keyboard::Equal){ // '+'
                    tempo_ms_ia = min(2000, tempo_ms_ia+100);
                }
                if(ev.key.code==sf::Keyboard::Hyphen || ev.key.code==sf::Keyboard::Subtract){ // '-'
                    tempo_ms_ia = max(60, tempo_ms_ia-50);
                }
            }

            if(jogoEncerrado) continue;

            if(ev.type==sf::Event::KeyPressed && ev.key.code==sf::Keyboard::P){
                if(estado.vez==BRANCO){
                    estado.aplicarJogada({-1,-1});
                    mostrarSug=false;
                    if(estado.passesSeguidos>=2) jogoEncerrado=true;
                }
            }
            if(ev.type==sf::Event::KeyPressed && ev.key.code==sf::Keyboard::S){
                if(estado.vez==BRANCO){
                    Estado tmp=estado; // mantém vez = BRANCO
                    MCTS aux; sugestao = aux.pensarTempo(tmp, max(80, tempo_ms_ia/2));
                    mostrarSug=true;
                }
            }
            if(ev.type==sf::Event::MouseButtonPressed && ev.mouseButton.button==sf::Mouse::Left){
                sf::Vector2f mp((float)ev.mouseButton.x, (float)ev.mouseButton.y);
                if(btPass.getGlobalBounds().contains(mp) && estado.vez==BRANCO){
                    estado.aplicarJogada({-1,-1});
                    mostrarSug=false;
                    if(estado.passesSeguidos>=2) jogoEncerrado=true;
                }else if(btSug.getGlobalBounds().contains(mp) && estado.vez==BRANCO){
                    Estado tmp=estado; MCTS aux; sugestao = aux.pensarTempo(tmp, max(80, tempo_ms_ia/2)); mostrarSug=true;
                }else if(L.dentroTab(mp.x, mp.y) && estado.vez==BRANCO){
                    auto [x,y] = L.posTab(mp);
                    sf::Vector2f alvo = L.posPixel(x,y);
                    float dx=mp.x-alvo.x, dy=mp.y-alvo.y;
                    if(std::sqrt(dx*dx+dy*dy) <= L.tol_click){
                        if(estado.movimentoLegal(BRANCO,x,y)){
                            estado.aplicarJogada({x,y});
                            mostrarSug=false;
                            if(estado.passesSeguidos>=2) jogoEncerrado=true;
                        }
                    }
                }
            }
        }

        // ============ Turno da IA (pretas) — assíncrono ============
        if(!jogoEncerrado && estado.vez==PRETO){
            if(!iaCalculando) startIA();
        }
        if(iaCalculando){
            using namespace std::chrono_literals;
            auto st = fut.wait_for(0ms);
            if(st==std::future_status::ready){
                Jogada j = fut.get();
                iaCalculando=false;
                estado.aplicarJogada(j);
                if(estado.passesSeguidos>=2) jogoEncerrado=true;
            }
        }

        // ============ Desenho ============
        win.clear(sf::Color(60,60,60));

        // fundo madeira
        sf::RectangleShape placa(sf::Vector2f((float)L.board, (float)L.board));
        placa.setPosition((float)L.baseX, (float)L.baseY);
        placa.setFillColor(sf::Color(240,210,140));
        win.draw(placa);

        // moldura
        sf::RectangleShape mold({(float)(L.board - 2*(L.margem-10)), (float)(L.board - 2*(L.margem-10))});
        mold.setPosition((float)(L.baseX + L.margem - 10), (float)(L.baseY + L.margem - 10));
        mold.setFillColor(sf::Color(0,0,0,0));
        mold.setOutlineColor(sf::Color(100,70,20,200));
        mold.setOutlineThickness(4.f);
        win.draw(mold);

        // grade
        sf::VertexArray linhas(sf::Lines);
        for(int i=0;i<TAM;i++){
            sf::Vector2f a = L.posPixel(0,i), b = L.posPixel(TAM-1,i);
            linhas.append(sf::Vertex(a, sf::Color::Black));
            linhas.append(sf::Vertex(b, sf::Color::Black));
            a = L.posPixel(i,0); b = L.posPixel(i,TAM-1);
            linhas.append(sf::Vertex(a, sf::Color::Black));
            linhas.append(sf::Vertex(b, sf::Color::Black));
        }
        win.draw(linhas);

        // hoshi
        static const vector<pair<int,int>> hoshi={{3,3},{9,3},{15,3},{3,9},{9,9},{15,9},{3,15},{9,15},{15,15}};
        for(auto [hx,hy]: hoshi){
            sf::CircleShape p(4.f); sf::Vector2f P=L.posPixel(hx,hy);
            p.setPosition(P.x-4.f, P.y-4.f); p.setFillColor(sf::Color::Black); win.draw(p);
        }

        // pedras nas INTERSEÇÕES
        for(int y=0;y<TAM;y++) for(int x=0;x<TAM;x++){
            int v=estado.get(x,y); if(v==VAZIO) continue;
            sf::CircleShape c(L.raio); sf::Vector2f P=L.posPixel(x,y);
            c.setPosition(P.x-L.raio, P.y-L.raio);
            c.setFillColor(v==PRETO?sf::Color(20,20,20):sf::Color(240,240,240));
            c.setOutlineThickness(2.f); c.setOutlineColor(sf::Color(0,0,0,120));
            win.draw(c);
        }

        // botões
        win.draw(btPass); win.draw(btSug);

        // textos
        auto placar = Estado::pontuarChines(estado);
        string vezStr = (estado.vez==PRETO? "Vez: Pretas (IA)" : "Vez: Brancas (Você)");
        drawText(vezStr, 16.f, (float)(L.H - L.ALT_UI + 14), 20);
        char buf[192];
        snprintf(buf,sizeof(buf),
                 "Capturas  P:%d  B:%d   |   Placar (chinês, komi=6.5)  P:%.1f  B:%.1f   |   Força IA: %d ms",
                 estado.captP, estado.captB, placar.first, placar.second, tempo_ms_ia);
        drawText(buf, 16.f, (float)(L.H - L.ALT_UI + 44), 18);
        drawText("Passar (P)", btPass.getPosition().x+14.f, btPass.getPosition().y+6.f, 18);
        drawText("Sugestão (S)", btSug.getPosition().x+8.f,  btSug.getPosition().y+6.f, 18);

        if(iaCalculando) drawText("IA pensando...", (float)(L.W - 260), (float)(L.baseY + 8), 18, sf::Color(40,40,200));
        if(mostrarSug && estado.vez==BRANCO && !sugestao.passe()){
            sf::CircleShape s(L.raio*0.45f); sf::Vector2f P=L.posPixel(sugestao.x,sugestao.y);
            s.setPosition(P.x - L.raio*0.45f, P.y - L.raio*0.45f);
            s.setFillColor(sf::Color(0,0,0,0));
            s.setOutlineColor(sf::Color(0,80,255,220));
            s.setOutlineThickness(3.f); win.draw(s);
        }

        if(jogoEncerrado){
            string fim = "Fim: dois passes. ";
            fim += (placar.first>placar.second? "Pretas vencem." : (placar.first<placar.second? "Brancas vencem." : "Empate."));
            drawText(fim, 16.f, (float)(L.baseY + 8), 22, sf::Color::Red);
        }

        win.display();
    }
    return 0;
}
