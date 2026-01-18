import { Dispatch, FC, SetStateAction, useState } from 'react';

import { IoIosCheckmark, IoIosRefresh } from 'react-icons/io';
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
    <>
      {/* TODO: Axel Refactor */}
      <div className="horizontal">
        <div>Current active learning model : </div>
        <Select
          options={availableModels}
          value={currentModel ? currentModel : null}
          onChange={(selectedOption) => {
            setCurrentModel(selectedOption ? selectedOption : null);
          }}
          isSearchable
          placeholder="Select a model for active learning"
        />
        <IoIosCheckmark
          size={40}
          style={{ color: 'green', cursor: 'pointer', margin: '0px 2px' }}
          onClick={() => setActiveQuickModel(currentModel)}
        />
        <PiEmptyBold
          size={20}
          style={{ color: 'red', cursor: 'pointer', margin: '0px 2px' }}
          onClick={() => {
            setCurrentModel(null);
            setActiveQuickModel(null);
          }}
        />
        {activeModel?.type === 'quickmodel' && (
          <div>
            Retrain now{' '}
            <IoIosRefresh
              size={20}
              style={{ color: 'green', cursor: 'pointer' }}
              onClick={() => {
                retrainQuickModel(activeModel.value);
                console.log('retrain');
              }}
            />
          </div>
        )}
      </div>
      {activeModel?.type === 'quickmodel' && (
        <div>
          <label htmlFor="frequencySlider" className="m-2">
            Retrain model every
          </label>
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
            style={{ flex: '1 1 30%', width: '60px' }}
          />
        </div>
      )}
    </>
  );
};
