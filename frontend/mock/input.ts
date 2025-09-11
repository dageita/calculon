// ./mock/users.ts

export default {

    // 返回值可以是数组形式
    'GET /llm_training_calculator/optimization_strategies': (req: any, res: any) => {
        res.send([
            "Full recomputation",
            "None recomputation",
            "Attention-only recomputation"
        ]);
    }
}