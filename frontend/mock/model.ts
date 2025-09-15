// ./mock/users.ts

export default {

    // 返回值可以是数组形式
    'GET /llm_training_calculator/model': (req: any, res: any) => {
        res.send([
            {
                "name": "LLAMA-7B",
                "seq_size": 4096,
                "hidden": 4096,
                "feedforward": 16384,
                "attn_heads": 32,
                "attn_size": 128,
                "num_blocks": 32,
                "vocab_size": 32000
            }
        ]);
    }
}