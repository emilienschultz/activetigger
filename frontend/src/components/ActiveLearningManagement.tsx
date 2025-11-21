import { Dispatch, FC, SetStateAction, useState } from 'react';

import { IoIosRefresh } from 'react-icons/io';
import { PiEmptyBold } from 'react-icons/pi';
import Select from 'react-select';
import { useRetrainQuickModel } from '../core/api';
import { AppContextValue } from '../core/context';
import { ModelDescriptionModel } from '../types';

/**
 * Component to manage one label
 */

interface ActiveLearningManagementProps {
  availableQuickModels: ModelDescriptionModel[];
  activeSimepleModel?: string | null;
  freqRefreshQuickModel?: number;
  projectName: string;
  currentScheme: string;
  setAppContext: Dispatch<SetStateAction<AppContextValue>>;
}

export const ActiveLearningManagement: FC<ActiveLearningManagementProps> = ({
  availableQuickModels,
  activeSimepleModel,
  freqRefreshQuickModel,
  setAppContext,
  projectName,
  currentScheme,
}) => {
  const [currentQuickModel, setCurrentQuickModel] = useState<string | null>(null);
  // function to change refresh frequency
  const refreshFreq = (newValue: number) => {
    setAppContext((prev) => ({ ...prev, freqRefreshQuickModel: newValue }));
  };
  const setActiveQuickModel = (newValue: string | null) => {
    setAppContext((prev) => ({ ...prev, activeQuickModel: newValue }));
  };

  // manage retrain of the model
  const { retrainQuickModel } = useRetrainQuickModel(projectName || null, currentScheme || null);

  return (
    <div>
      <div>
        Current active learning model :{' '}
        <b>
          {activeSimepleModel ? (
            <span>
              {activeSimepleModel}
              <PiEmptyBold
                className="mx-2"
                size={20}
                style={{ color: 'red', cursor: 'pointer' }}
                onClick={() => setActiveQuickModel(null)}
              />
              <IoIosRefresh
                size={20}
                style={{ color: 'green', cursor: 'pointer' }}
                onClick={() => {
                  retrainQuickModel(activeSimepleModel || '');
                  console.log('retrain');
                }}
              />
            </span>
          ) : (
            'no model selected'
          )}
        </b>
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
          step="5"
          className="mx-2"
        />
        annotations (0 for no refreshing)
      </div>
    </div>
  );
};
