// SPDX-License-Identifier: MIT
// memory_palace_pmll_llama.cpp
//
// GPT-5 “Persistent Memory Logic Loop” for llama.cpp
// © 2025 Dr. Josef Kurk Edwards & John Trompeter
//
// One-file amalgamation: PMLL Loop  + Memory Palace/Matrix + Hook + GPT-5 façade
// Build: g++ -std=c++17 -I./include -I/path/to/graphiti -I/path/to/json/include \
//            memory_palace_pmll_llama.cpp -o demo -lllama -pthread

//-----------------------------------  INCLUDES  --------------------------------
#include "llama.h"                     // llama.cpp public C API
#include <graphiti/graph.hpp>          // Graphiti knowledge graph
#include <nlohmann/json.hpp>           // single-header JSON
#include <filesystem>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <algorithm>
#include <cmath>
#include <iostream>

//-----------------------------------  NAMESPACES  ------------------------------
namespace pmll {                         // forward declarations
struct  LoopHook;
class   Loop;
}

namespace pmll::memory {
//-----------------------------------  ENUMS / HELPERS  -------------------------
enum class ConceptType { FACTUAL, PROCEDURAL, EPISODIC, SEMANTIC,
                         WORKING, LONG_TERM, ASSOCIATIVE };

inline std::string concept_type_to_string(ConceptType t) {
    switch (t) {
        case ConceptType::FACTUAL:      return "FACTUAL";
        case ConceptType::PROCEDURAL:   return "PROCEDURAL";
        case ConceptType::EPISODIC:     return "EPISODIC";
        case ConceptType::SEMANTIC:     return "SEMANTIC";
        case ConceptType::WORKING:      return "WORKING";
        case ConceptType::LONG_TERM:    return "LONG_TERM";
        case ConceptType::ASSOCIATIVE:  return "ASSOCIATIVE";
    } return "UNKNOWN";
}

//-----------------------------------  MEMORY PALACE  ---------------------------
class MemoryPalace {
  public:
    struct Room {
        std::string name;
        ConceptType type{};
        std::vector<std::string> node_ids;
        nlohmann::json spatial;
    };

    explicit MemoryPalace(const std::string&) { init_defaults(); }
    void add_room(const std::string& n, ConceptType t) { rooms_[n] = {n,t}; }
    void assign(const std::string& id, const std::string& room) {
        rooms_[room].node_ids.push_back(id);
    }
    std::vector<std::string> contents(const std::string& room) const {
        auto it = rooms_.find(room);
        return it==rooms_.end()?std::vector<std::string>{}:it->second.node_ids;
    }
  private:
    std::unordered_map<std::string,Room> rooms_;
    void init_defaults() {
        add_room("facts_library",   ConceptType::FACTUAL);
        add_room("skills_workshop", ConceptType::PROCEDURAL);
        add_room("conversation_hall",ConceptType::EPISODIC);
        add_room("concepts_tower",  ConceptType::SEMANTIC);
        add_room("working_desk",    ConceptType::WORKING);
        add_room("archive_vault",   ConceptType::LONG_TERM);
        add_room("connection_web",  ConceptType::ASSOCIATIVE);
    }
};

//-----------------------------------  NODE / EDGE  -----------------------------
using Clock = std::chrono::system_clock;

struct MemoryWeight {
    float importance{0.5f}, recency{1.0f}, frequency{0.0f}, coherence{0.5f};
    float composite() const {
        return importance*0.4f + recency*0.2f + frequency*0.2f + coherence*0.2f;
    }
};

struct MemoryNode {
    std::string id;
    ConceptType type{ConceptType::SEMANTIC};
    std::string content;
    std::vector<float> emb;
    MemoryWeight w;
    Clock::time_point created{Clock::now()}, last{Clock::now()};
    uint32_t hits{0};
    nlohmann::json meta;

    void touch() { last = Clock::now(); ++hits; 
        auto hrs = std::chrono::duration_cast<std::chrono::hours>
                   (Clock::now()-last).count();
        w.recency = std::exp(-hrs/168.f);
    }
};

struct MemoryEdge {
    std::string from,to,rel; float strength{0.5f}; nlohmann::json ctx;
};

//-----------------------------------  MEMORY MATRIX  ---------------------------
class MemoryMatrix {
  public:
    MemoryMatrix(const std::string& gpath,const std::string& ppath,
                 std::size_t maxw=1000)
      : graph_(std::make_unique<graphiti::Graph>(gpath)),
        palace_(ppath), max_working_(maxw) {}

    std::string store(const std::string& txt, ConceptType t=ConceptType::SEMANTIC,
                      const nlohmann::json& meta={}) {
        auto id=new_id(); auto e=embed(txt);
        MemoryNode n{id,t,txt,e,MemoryWeight{},Clock::now(),Clock::now(),0,meta};
        nodes_[id]=n; graph_->add_node(id,txt,e); assign_room(id,t);
        auto_link(id); trim_working(); return id;
    }

    std::vector<MemoryNode> retrieve(const std::string& q,std::size_t k=10,
                                     float th=0.7f) {
        std::vector<MemoryNode> out;
        for(auto [id,s] : graph_->similarity_search(embed(q),k)){
            if(s<th||!nodes_.count(id)) continue;
            nodes_[id].touch(); out.push_back(nodes_[id]);
        }
        std::sort(out.begin(),out.end(),
                  [](auto&a,auto&b){return a.w.composite()>b.w.composite();});
        return out;
    }

    void create_edge(const std::string&a,const std::string&b,
                     const std::string&r,float s=0.5f){
        edges_.push_back({a,b,r,s,{}});
        graph_->add_edge(a,b,r,s);
    }

    void consolidate(){ strengthen(); decay(); /* TODO: community merge */ }

  private:
    std::unique_ptr<graphiti::Graph> graph_;
    MemoryPalace palace_;
    std::unordered_map<std::string,MemoryNode> nodes_;
    std::vector<MemoryEdge> edges_;
    std::size_t max_working_;

    //---------------- helpers -------------------------------------------------
    static std::vector<float> embed(const std::string& t){
        constexpr std::size_t D=384; std::vector<float> v(D);
        std::hash<std::string> H; auto h=H(t);
        for(std::size_t i=0;i<D;++i) v[i]=((h>>(i%32))&1)?1.f:-1.f;
        return v;
    }
    static std::string timestamp(){
        return std::to_string(std::chrono::duration_cast<
               std::chrono::milliseconds>(Clock::now().time_since_epoch()).count());
    }
    std::string new_id(){ static uint64_t c=0; return "node_"+timestamp()+"_"+std::to_string(++c); }

    void assign_room(const std::string& id,ConceptType t){
        static const std::unordered_map<ConceptType,std::string> map={
          {ConceptType::FACTUAL,"facts_library"},
          {ConceptType::PROCEDURAL,"skills_workshop"},
          {ConceptType::EPISODIC,"conversation_hall"},
          {ConceptType::SEMANTIC,"concepts_tower"},
          {ConceptType::WORKING,"working_desk"},
          {ConceptType::LONG_TERM,"archive_vault"},
          {ConceptType::ASSOCIATIVE,"connection_web"}};
        palace_.assign(id,map.at(t));
    }
    void auto_link(const std::string& id){
        for(auto [o,s]:graph_->similarity_search(nodes_[id].emb,5))
            if(o!=id&&s>0.8f) create_edge(id,o,"similar_to",s);
    }
    void trim_working(){
        auto ws=palace_.contents("working_desk");
        if(ws.size()<=max_working_) return;
        std::vector<std::pair<std::string,float>> v;
        for(auto& id:ws) v.emplace_back(id,nodes_[id].w.composite());
        std::sort(v.begin(),v.end(),[](auto&a,auto&b){return a.second<b.second;});
        std::size_t n=ws.size()*0.2;
        for(std::size_t i=0;i<n && i<v.size(); ++i){
            auto& node=nodes_[v[i].first]; node.type=ConceptType::LONG_TERM;
            assign_room(node.id,ConceptType::LONG_TERM);
        }
    }
    void strengthen(){
        for(auto& e:edges_) if(nodes_[e.from].hits+nodes_[e.to].hits>20) e.strength*=1.1f;
    }
    void decay(){
        for(auto& [id,n]:nodes_) n.w.recency*=0.98f;
    }
};

//-----------------------------------  PMLL LOOP  ------------------------------
} // namespace pmll::memory

namespace pmll {

// Thin logic-loop callback interface
struct LoopHook {
    virtual bool operator()(const std::string& prompt,
                            const std::vector<llama_token>& last_out)=0;
    virtual ~LoopHook()=default;
};

// Persistent KV-snapshot loop for llama.cpp
class Loop {
  public:
    Loop(const std::string& model_path,const std::string& state_dir,
         uint32_t n_ctx=4096, LoopHook* hook=nullptr)
      : model_path_(model_path),state_dir_(state_dir),hook_(hook) {

        std::filesystem::create_directories(state_dir_);
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        model_=llama_model_load_from_file(model_path.c_str(),mp);
        if(!model_) throw std::runtime_error("model load failed");

        llama_context_params cp=llama_context_default_params(); cp.n_ctx=n_ctx;
        ctx_=llama_init_from_model(model_,cp);
        if(!ctx_) throw std::runtime_error("ctx fail");
    }
    ~Loop(){ llama_free(ctx_); llama_model_free(model_); llama_backend_free(); }

    std::string generate(const std::string& prompt,int n_predict=128,
                         llama_seq_id seq=0){
        std::lock_guard<std::mutex> lock(mu_);
        restore(seq);

        // tokenize prompt
        std::vector<llama_token> toks(prompt.size()+8);
        int n = llama_tokenize(model_,prompt.c_str(),toks.data(),toks.size(),true,true);
        toks.resize(n);

        llama_batch b = llama_batch_init(n,0,1);
        for(int i=0;i<n;++i){ b.token[i]=toks[i]; b.pos[i]=i; b.seq_id[i]=&seq; b.n_seq_id[i]=1; }
        llama_decode(ctx_,b); llama_batch_free(b);

        std::vector<llama_token> out; out.reserve(n_predict);
        for(int step=0; step<n_predict; ++step){
            llama_batch d = llama_batch_init(1,0,1);
            d.token[0]=sample_next(); d.pos[0]=toks.size()+step;
            d.seq_id[0]=&seq; d.n_seq_id[0]=1;
            llama_decode(ctx_,d); out.push_back(d.token[0]); llama_batch_free(d);

            if(hook_ && !(*hook_)(prompt,out)) break;
            persist(seq);
            if(out.back()==llama_token_eos()) break;
        }
        return tokens_to_str(out);
    }

  private:
    //---------------- helpers -------------------------------------------------
    llama_token sample_next(){
        const float* l=llama_get_logits(ctx_);
        int n_vocab=llama_n_vocab(llama_model_get_vocab(model_));
        return std::max_element(l,l+n_vocab)-l; // greedy
    }
    void persist(llama_seq_id seq){
        std::string f=state_dir_+"/seq-"+std::to_string(seq)+".pmll";
        llama_state_seq_save_file(ctx_,f.c_str(),seq,nullptr,0);
    }
    void restore(llama_seq_id seq){
        std::string f=state_dir_+"/seq-"+std::to_string(seq)+".pmll";
        if(std::filesystem::exists(f))
            llama_state_seq_load_file(ctx_,f.c_str(),seq,nullptr,0,nullptr);
    }
    std::string tokens_to_str(const std::vector<llama_token>& t){
        std::string s; char buf[8];
        for(auto tok:t){ int n=llama_token_to_str(model_,tok,buf,8);
            if(n>0) s.append(buf,n);}
        return s;
    }

    std::mutex mu_; std::string model_path_,state_dir_; LoopHook* hook_;
    llama_model* model_{nullptr}; llama_context* ctx_{nullptr};
};

} // namespace pmll

//-----------------------------------  MEMORY HOOK  ----------------------------
namespace pmll::memory {

class MemoryHook : public pmll::LoopHook {
  public:
    MemoryHook(MemoryMatrix& m,const std::string& sess)
      : mx_(m),sess_(sess) {}

    bool operator()(const std::string& prompt,
                    const std::vector<llama_token>& out) override {
        // flush buffer every 40 tokens or on EOS
        if(!out.empty()){
            auto tok=out.back();
            if(tok==llama_token_eos()||out.size()%40==0){
                if(!buffer_.empty()){
                    mx_.store(buffer_,ConceptType::EPISODIC,{{"session",sess_}});
                    buffer_.clear();
                }
            } else{
                char buf[8]; int n=llama_token_to_str(nullptr,tok,buf,8);
                if(n>0) buffer_.append(buf,n);
            }
        }
        return true;
    }
  private:
    MemoryMatrix& mx_; std::string sess_; std::string buffer_;
};

//-----------------------------------  GPT-5 SYSTEM  ---------------------------
class GPT5MemorySystem {
  public:
    GPT5MemorySystem(const std::string& model,const std::string& memdir,
                     const std::string& sess="default")
      : mx_(memdir+"/graph",memdir+"/palace"),
        hook_(mx_,sess),
        loop_(model,memdir+"/llama_state",8192,&hook_) {}

    std::string chat(const std::string& prompt,int max=256){
        auto ctx = mx_.retrieve(prompt,8);
        std::string ep="[MEMORIES]\n";
        for(auto& n:ctx) ep+="• "+n.content+"\n";
        ep+="\n[USER] "+prompt+"\n[ASSISTANT]";
        auto rsp = loop_.generate(ep,max);
        mx_.store(prompt+"\n"+rsp,ConceptType::EPISODIC);
        if(++turn_%10==0) mx_.consolidate();
        return rsp;
    }
  private:
    MemoryMatrix mx_; MemoryHook hook_; pmll::Loop loop_; int turn_{0};
};

} // namespace pmll::memory

//-----------------------------------  OPTIONAL DEMO  --------------------------
#ifdef PMLL_DEMO_MAIN
int main() {
    pmll::memory::GPT5MemorySystem sys(
        "./models/Llama-3-8B-Instruct.gguf","./state","demo");
    std::cout << sys.chat("Explain the PMLL architecture.") << "\n";
}
#endif
