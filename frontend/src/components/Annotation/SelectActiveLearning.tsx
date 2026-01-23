import { FC, useEffect, useMemo, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { useRetrainQuickModel, useTrainQuickModel } from '../../core/api';
import { useAppContext } from '../../core/context';
import { useNotifications } from '../../core/notifications';
import { ActiveLearningManagement } from '../ActiveLearningManagement';

interface SelectActiveLearningProps {
  display: boolean;
  setActiveMenu: (value: boolean) => void;
  setSelectFirstModelTrained?: (value: boolean) => void;
  selectFirstModelTrained?: boolean;
  numberAnnotated?: number;
}

export const SelectActiveLearning: FC<SelectActiveLearningProps> = ({
  display,
  setActiveMenu,
  setSelectFirstModelTrained,
  selectFirstModelTrained,
  numberAnnotated = 0,
}) => {
  const { notify } = useNotifications();

  const {
    appContext: { freqRefreshQuickModel, activeModel, currentScheme, currentProject: project },
    setAppContext,
  } = useAppContext();

  const projectName = project?.params.project_name;

  // existing models
  const availableQuickModels = useMemo(
    () => project?.quickmodel.available[currentScheme || ''] || [],
    [project?.quickmodel, currentScheme],
  );
  const availableBertModels = useMemo(
    () => project?.languagemodels.available[currentScheme || ''] || {},
    [project?.languagemodels, currentScheme],
  );

  const availableBertModelsWithPrediction = Object.entries(availableBertModels || {})
    .filter(([_, v]) => v && v.predicted)
    .map(([k, _]) => k);

  // TODO only keep those with prediction
  const groupedModels = [
    {
      label: 'Quick Models',
      options: (availableQuickModels ?? [])
        .filter((e) => e?.name) // <-- protect against undefined/missing name
        .map((e) => ({
          value: e.name,
          label: e.name,
          type: 'quickmodel',
        })),
    },
    {
      label: 'Language Models',
      options: (availableBertModelsWithPrediction ?? [])
        .filter((e) => e) // <-- ensure non-null
        .map((e) => ({
          value: e,
          label: e,
          type: 'languagemodel',
        })),
    },
  ];

  const { trainQuickModel } = useTrainQuickModel(projectName || null, currentScheme || null);
  const startTrainQuickModel = () => {
    // default quickmodel
    const availableFeatures = project?.features.available ? project?.features.available : [];
    if (availableFeatures.length === 0) {
      setActiveMenu(false);
      notify({
        type: 'warning',
        message: 'No features available for quickmodel',
      });
    }
    const formData = {
      name: 'basic-quickmodel',
      model: 'logistic-l1',
      scheme: currentScheme || '',
      params: {
        costLogL2: 1,
        costLogL1: 1,
        n_neighbors: 3,
        alpha: 1,
        n_estimators: 500,
        max_features: null,
      },
      dichotomize: null,
      features: availableFeatures,
      cv10: false,
      standardize: false,
      balance_classes: false,
    };
    trainQuickModel(formData);
    setActiveMenu(false);
    if (setSelectFirstModelTrained) setSelectFirstModelTrained(true);
  };

  // deactivate active model if it has been removed from available models
  useEffect(() => {
    if (
      activeModel &&
      !availableQuickModels.find((model) => model.name === activeModel.value) &&
      activeModel.type === 'quickmodel'
    ) {
      setAppContext((prev) => ({ ...prev, activeModel: null }));
    }
    if (
      activeModel &&
      !Object.keys(availableBertModels).includes(activeModel.value) &&
      activeModel.type === 'languagemodel'
    ) {
      setAppContext((prev) => ({ ...prev, activeModel: null }));
    }
  }, [availableQuickModels, activeModel, setAppContext, availableBertModels]);

  // fastrack active learning model
  useEffect(() => {
    if (selectFirstModelTrained && availableQuickModels.length > 0) {
      // select the first trained model
      setAppContext((prev) => ({
        ...prev,
        activeModel: {
          type: 'quickmodel',
          value: availableQuickModels[0].name,
          label: availableQuickModels[0].name,
        },
        selectionConfig: { ...prev.selectionConfig, mode: 'active' },
      }));
    }
  }, [availableQuickModels, selectFirstModelTrained, setAppContext]);

  // retrain quick model (only for )
  const { retrainQuickModel } = useRetrainQuickModel(projectName || null, currentScheme || null);
  const [updatedQuickModel, setUpdatedQuickModel] = useState(false);
  useEffect(() => {
    // only the training points for the current phase

    if (
      !updatedQuickModel &&
      freqRefreshQuickModel &&
      activeModel &&
      numberAnnotated > 0 &&
      numberAnnotated % freqRefreshQuickModel == 0 &&
      activeModel.type === 'quickmodel'
    ) {
      setUpdatedQuickModel(true);
      retrainQuickModel(activeModel.value);
    }
    if (
      updatedQuickModel &&
      freqRefreshQuickModel &&
      numberAnnotated % freqRefreshQuickModel != 0
    ) {
      setUpdatedQuickModel(false);
    }
  }, [
    freqRefreshQuickModel,
    setUpdatedQuickModel,
    activeModel,
    updatedQuickModel,
    retrainQuickModel,
    numberAnnotated,
    projectName,
  ]);

  return (
    <Modal show={display} onHide={() => setActiveMenu(false)} id="active-modal" size="lg">
      <Modal.Header closeButton>
        <Modal.Title>Configure active learning</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        {availableQuickModels.length + Object.keys(availableBertModels).length > 0 ? (
          <ActiveLearningManagement
            availableModels={groupedModels}
            setAppContext={setAppContext}
            freqRefreshQuickModel={freqRefreshQuickModel}
            activeModel={activeModel}
            projectName={projectName || ''}
            currentScheme={currentScheme || ''}
          />
        ) : (
          <>
            <div className="horizontal center">
              No quick model currently available. Go to model tab or
            </div>
            <button className="btn-submit" onClick={startTrainQuickModel}>
              Train a default quick model
            </button>
          </>
        )}
      </Modal.Body>
    </Modal>
  );
};
