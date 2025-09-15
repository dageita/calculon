// ./mock/users.ts

export default {

    // 返回值可以是数组形式
    'GET /llm_training_calculator/gpu': (req: any, res: any) => {
        res.send([
            {
                "name": "H200",
                "sparse_tensor_fp16_processing_power": 989,
                "sparse_tensor_fp32_processing_power": 495,
                "memory": 141,
                "memory_bandwidth": 4900,
                "bus_bandwidth": 900,
                "support_p2p": true
            }
        ]);
    },

    // 返回值也可以是对象形式
    'GET /llm_training_calculator/network': (req: any, res: any) => {
        res.send({
            "network_bandwidth": 0,
            "network_topology": ["Single machine", "One big switch", "Spine-leaf"]
        })
    },

}