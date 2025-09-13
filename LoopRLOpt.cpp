// LoopRLOpt.cpp
// Build as an LLVM pass to extract loop features and optionally apply vectorization metadata.
//
// Minimal tested with LLVM 10..16 APIs; you may need small API changes for other versions.

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/JSON.h"
#include <fstream>
#include <nlohmann/json.hpp> // optional: you'd need a single-header JSON or write manually

using namespace llvm;

namespace {
    struct LoopRLOpt : public ModulePass {
        static char ID;
        std::string OutputJSON = "loop_features.json";
        // Optional: pass-in actions JSON filename via a module flag or environment var.
        std::string ActionsJSON = "loop_actions.json";

        LoopRLOpt() : ModulePass(ID) {}

        bool runOnModule(Module& M) override {
            // Get analyses we need per-function in runOnFunction style; legacy ModulePass will query per-Fn.
            std::vector<nlohmann::json> loopsJson;

            // Acquire ScalarEvolution and LoopInfo via analysis managers per function:
            for (Function& F : M) {
                if (F.isDeclaration()) continue;
                DominatorTree DT(F);
                LoopInfo LI;
                LI.analyze(DT);
                // Note: ScalarEvolution requires more setup; here we attempt a conservative approach.
                for (Loop* L : LI) {
                    nlohmann::json j;
                    j["function"] = std::string(F.getName());
                    j["header"] = std::string(L->getHeader()->getName());
                    // estimate trip count if SCEV available
                    long long tripCount = -1;
                    if (L->getLoopPreheader()) {
                        // naive: check for canonical induction variable with constant bound (best-effort)
                        // Full SCEV approach would use ScalarEvolution.
                    }
                    j["trip_count_est"] = tripCount;
                    // count loads / stores / arithmetic
                    unsigned loads = 0, stores = 0, arith = 0, calls = 0;
                    for (BasicBlock* BB : L->blocks()) {
                        for (Instruction& I : *BB) {
                            if (isa<LoadInst>(I)) loads++;
                            else if (isa<StoreInst>(I)) stores++;
                            else if (isa<CallInst>(I) || isa<InvokeInst>(I)) calls++;
                            else if (I.isBinaryOp()) arith++;
                        }
                    }
                    j["num_loads"] = loads;
                    j["num_stores"] = stores;
                    j["num_arith"] = arith;
                    j["num_calls"] = calls;

                    // control dependence: number of basic blocks in loop
                    j["num_blocks"] = L->getNumBlocks();

                    // Identify a unique loop id using function + header ptr
                    std::string loop_id = std::string(F.getName()) + ":" + L->getHeader()->getName().str();
                    j["loop_id"] = loop_id;

                    loopsJson.push_back(j);
                }
            }

            // write JSON to disk
            nlohmann::json out = loopsJson;
            std::ofstream ofs(OutputJSON);
            ofs << out.dump(2);
            ofs.close();

            // Next: if actions file present, read it and apply metadata
            std::ifstream ifs(ActionsJSON);
            if (ifs.good()) {
                nlohmann::json acts = nlohmann::json::parse(ifs);
                applyActions(M, acts);
            }

            return true;
        }

        void applyActions(Module& M, nlohmann::json& acts) {
            // acts is expected to be a map: loop_id -> action { "type":"width", "value":4 } or "disable":true
            for (Function& F : M) {
                if (F.isDeclaration()) continue;
                for (BasicBlock& BB : F) {
                    // look for headers matching loop ids from actions
                    // This simplistic approach matches block names used earlier.
                    if (acts.is_null()) return;
                }
            }

            // A better approach is to iterate loops again and match loop header names:
            for (Function& F : M) {
                if (F.isDeclaration()) continue;
                DominatorTree DT(F);
                LoopInfo LI;
                LI.analyze(DT);
                for (Loop* L : LI) {
                    std::string loop_id = std::string(F.getName()) + ":" + L->getHeader()->getName().str();
                    if (acts.find(loop_id) == acts.end()) continue;
                    nlohmann::json a = acts[loop_id];
                    // Build metadata:
                    LLVMContext& Ctx = M.getContext();
                    SmallVector<Metadata*, 4> MDs;
                    // existing llvm.loop MD should start with self reference; using simplified method:
                    if (a.contains("disable") && a["disable"].get<bool>() == true) {
                        // attach metadata: !llvm.loop.vectorize.enable = !0 (false)
                        MDs.push_back(MDString::get(Ctx, "llvm.loop.vectorize.enable"));
                        MDs.push_back(ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 0)));
                    }
                    else if (a.contains("width")) {
                        int width = a["width"].get<int>();
                        // attach llvm.loop.vectorize.width = <width>
                        MDs.push_back(MDString::get(Ctx, "llvm.loop.vectorize.width"));
                        MDs.push_back(ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), width)));
                    }

                    if (!MDs.empty()) {
                        MDNode* Node = MDNode::get(Ctx, MDs);
                        // Attach to the header's terminator or a representative instruction
                        Instruction& I = L->getHeader()->front();
                        I.setMetadata("llvm.loop", Node);
                    }
                }
            }
        }
    }; // end struct

    char LoopRLOpt::ID = 0;
    static RegisterPass<LoopRLOpt> X("loop-rl-opt", "Loop RL Extract/Apply Pass", false, false);
} // namespace
