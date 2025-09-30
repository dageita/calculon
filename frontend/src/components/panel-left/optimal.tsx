import {
    Select,
    InputNumber,
    Slider,
    Popover
} from 'antd';
import useModel from 'flooks';
import styles from './index.less';
import ProjectModel from '@/models/projectModel';
import { InfoCircleOutlined } from '@ant-design/icons';
import LogModel from '@/models/logModel';
import { getDataTypes } from '@/services';
import { useImmer } from 'use-immer';
import { useEffect } from 'react';
import { useTranslation } from 'react-i18next';


const OptimalPanel = (props) => {
    // 从ProjectModel中获取curCpu状态
    const { setProject, setOtherConfig, otherConfig, recommendConfig, curModel, curGpu,
        curCouHasChanged, checkPipeline, curCpu } = useModel(ProjectModel);
    const { t } = useTranslation();
    const { setChangeLog } = useModel(LogModel);

    const [state, setState] = useImmer({
        dataTypeList: [],
        lastGpuValue: null,
    });

    // 设置参数值并记录变更日志
    const setParamValue = (key, val, title) => {
        setChangeLog(title, val, otherConfig?.[key]);
        setOtherConfig({ [key]: val });
    };


    // 加载优化策略列表
    const loadDataTypes = async () => {
        // 注意：这里原代码使用了curGpu.value，可能需要根据实际情况调整
        const result = await getDataTypes(curGpu.value) as any;
        const dataTypes = result.datatypes.map((item: any) => ({
            key: item,
            label: item,
            value: item,
        }));
        setState(prev => ({
            ...prev,
            dataTypeList: dataTypes
        }));
    };

    // 组件卸载时记录当前GPU值
    useEffect(() => {
        return () => {
            // 组件卸载时保存当前GPU值
            if (curGpu?.value) {
                setState(prev => ({
                    ...prev,
                    lastGpuValue: curGpu.value
                }));
            }
        };
    }, [curGpu?.value]); // 依赖curGpu.value，确保值变化时能正确记录

    // 组件挂载时比较GPU值
    useEffect(() => {
        // 组件挂载时，如果存在上次记录的GPU值，则进行比较
        if (state.lastGpuValue !== null && curGpu?.value) {
            const isSame = state.lastGpuValue === curGpu.value;
            // 这里可以根据实际需求处理比较结果
            console.log(`GPU值是否与上次相同: ${isSame}`);
            // 如果需要可以添加其他逻辑，比如触发某些函数
            if (!isSame) {

                if (otherConfig?.datatype) {
                    setParamValue('datatype', undefined, 'Data Type');
                }
            }
        }
    }, []); // 空依赖数组表示只在组件挂载时执行一次



    // 组件加载时获取策略列表
    useEffect(() => {
        loadDataTypes();
    }, []);


    // 渲染微批次大小设置组件
    const renderBatchSize = () => (
        <div>
            {recommendConfig.recomended_microbatch && (
                <div className={styles.slider_tip}>
                    {t('microbatch recommend', { value: recommendConfig.recomended_microbatch })}
                </div>
            )}
            <InputNumber
                className={styles.number_item}
                precision={0}
                min={1}
                value={otherConfig?.max_batch_size}
                onChange={(val) => setParamValue('max_batch_size', val, 'Max Batch size')}
                addonAfter={(
                    <Popover content={<div>Need to be able to divide data parallel degree.</div>}>
                        <InfoCircleOutlined style={{ cursor: 'pointer' }} />
                    </Popover>
                )}
            />
        </div>
    );

    return (
        <div className={styles.nest}>
            <p className={styles.section_title}>{t('Gpu Numbers')}</p>
            <div className={styles['group-content']}>
                <InputNumber
                    className={styles['input-num-content']}
                    min={1}
                    disabled
                    value={curGpu?.num_procs}
                    onChange={(val: any) => {
                        setProject({
                            curGpu: {
                                ...curGpu,
                                num_procs: val
                            }
                        });
                    }}
                />
            </div>

            <p className={styles.section_title}>{t('Max Batch Size')}</p>
            <div className={styles.section_content}>
                {renderBatchSize()}
            </div>

            <p className={styles.section_title}>{t('datatype')}</p>
            <div className={styles['group-content']}>
                <Select
                    options={state.dataTypeList}
                    placeholder={t('Select one datatype')}
                    value={otherConfig['datatype']}
                    onChange={(val) => setParamValue('datatype', val, 'Data Type')}
                />
            </div>
        </div>
    );
};

export default OptimalPanel;
