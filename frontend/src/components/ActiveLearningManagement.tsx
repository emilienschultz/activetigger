import { Dispatch, FC, SetStateAction, useEffect, useState } from 'react';

import Select from 'react-select';
import { useRetrainQuickModel } from '../core/api';
import { AppContextValue } from '../core/context';
import { ModelDescriptionModel } from '../types';

/**
 * Component to manage one label
 */

interface ActiveLearningManagementProps {
  projectSlug: string | null;
  currentScheme: string | null;
  history: string[];
  availableQuickModels: ModelDescriptionModel[];
  activeSimepleModel?: string | null;
  freqRefreshQuickModel?: number;
  setAppContext: Dispatch<SetStateAction<AppContextValue>>;
}

export const ActiveLearningManagement: FC<ActiveLearningManagementProps> = ({
  projectSlug,
  currentScheme,
  availableQuickModels,
  activeSimepleModel,
  freqRefreshQuickModel,
  history,
  setAppContext,
}) => {
  const [currentQuickModel, setCurrentQuickModel] = useState<string | null>(null);
  // function to change refresh frequency
  const refreshFreq = (newValue: number) => {
    setAppContext((prev) => ({ ...prev, freqRefreshQuickModel: newValue }));
  };
  const setActiveQuickModel = (newValue: string | null) => {
    setAppContext((prev) => ({ ...prev, activeQuickModel: newValue }));
  };
  const { retrainQuickModel } = useRetrainQuickModel(projectSlug, currentScheme);

  // manage retrain of the model
  const [updatedQuickModel, setUpdatedQuickModel] = useState(false);
  useEffect(() => {
    if (
      !updatedQuickModel &&
      freqRefreshQuickModel &&
      activeSimepleModel &&
      history.length > 0 &&
      history.length % freqRefreshQuickModel == 0
    ) {
      setUpdatedQuickModel(true);
      retrainQuickModel(activeSimepleModel);
    }
    if (updatedQuickModel && freqRefreshQuickModel && history.length % freqRefreshQuickModel != 0) {
      setUpdatedQuickModel(false);
    }
  }, [
    freqRefreshQuickModel,
    setUpdatedQuickModel,
    activeSimepleModel,
    updatedQuickModel,
    retrainQuickModel,
    history,
  ]);

  return (
    <div>
      <div>
        Current active learning model :{' '}
        <b>{activeSimepleModel ? activeSimepleModel : 'no model selected'}</b>
      </div>
      <div>
        <div className="d-flex align-items-center my-2">
          <Select
            options={Object.values(availableQuickModels || {}).map((e) => ({
              value: e.name,
              label: e.name,
            }))}
            value={
              currentQuickModel ? { value: currentQuickModel, label: currentQuickModel } : null
            }
            onChange={(selectedOption) => {
              setCurrentQuickModel(selectedOption ? selectedOption.value : null);
            }}
            isSearchable
            placeholder="Select a model for active learning"
          />
          <button
            className="btn btn-primary mx-2"
            onClick={() => setActiveQuickModel(currentQuickModel)}
          >
            Select
          </button>
        </div>
      </div>
      <div className="d-flex align-items-center">
        <label htmlFor="frequencySlider">Retrain model every</label>
        <input
          type="number"
          id="frequencySlider"
          min="0"
          max="500"
          value={freqRefreshQuickModel}
          onChange={(e) => {
            refreshFreq(Number(e.currentTarget.value));
          }}
          step="1"
          className="mx-2"
        />
        annotations (0 for no refreshing)
      </div>
    </div>
  );
};
