import { Dispatch, FC, SetStateAction, useState } from 'react';

import { IoIosRefresh } from 'react-icons/io';
import { PiEmptyBold } from 'react-icons/pi';
import Select from 'react-select';
import { useRetrainQuickModel } from '../core/api';
import { AppContextValue } from '../core/context';
import { ActiveModel } from '../types';

/**
 * Component to manage one label
 */

interface ActiveLearningManagementProps {
  availableModels: {
    label: string;
    options: ActiveModel[];
  }[];
  activeModel?: ActiveModel | null;
  freqRefreshQuickModel?: number;
  projectName: string;
  currentScheme: string;
  setAppContext: Dispatch<SetStateAction<AppContextValue>>;
}

export const ActiveLearningManagement: FC<ActiveLearningManagementProps> = ({
  availableModels,
  activeModel,
  freqRefreshQuickModel,
  setAppContext,
  projectName,
  currentScheme,
}) => {
  const [currentModel, setCurrentModel] = useState<ActiveModel | null>(null);
  // function to change refresh frequency
  const refreshFreq = (newValue: number) => {
    setAppContext((prev) => ({ ...prev, freqRefreshQuickModel: newValue }));
  };
  const setActiveQuickModel = (newValue: ActiveModel | null) => {
    setAppContext((prev) => ({ ...prev, activeModel: newValue }));
  };

  // manage retrain of the model
  const { retrainQuickModel } = useRetrainQuickModel(projectName || null, currentScheme || null);

  return (
    <div>
      <div>
        Current active learning model :{' '}
        <b>
          {activeModel ? (
            <span>
              {activeModel.value}
              <PiEmptyBold
                className="mx-2"
                size={20}
                style={{ color: 'red', cursor: 'pointer' }}
                onClick={() => setActiveQuickModel(null)}
              />
              {activeModel.type === 'quickmodel' && (
                <IoIosRefresh
                  size={20}
                  style={{ color: 'green', cursor: 'pointer' }}
                  onClick={() => {
                    retrainQuickModel(activeModel.value);
                    console.log('retrain');
                  }}
                />
              )}
            </span>
          ) : (
            'no model selected'
          )}
        </b>
      </div>
      <div>
        <div className="d-flex align-items-center my-2">
          <Select
            options={availableModels}
            value={currentModel ? currentModel : null}
            onChange={(selectedOption) => {
              setCurrentModel(selectedOption ? selectedOption : null);
            }}
            isSearchable
            placeholder="Select a model for active learning"
          />
          <button
            className="btn btn-primary mx-2"
            onClick={() => setActiveQuickModel(currentModel)}
          >
            Select
          </button>
        </div>
      </div>
      {activeModel?.type === 'quickmodel' && (
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
      )}
    </div>
  );
};
