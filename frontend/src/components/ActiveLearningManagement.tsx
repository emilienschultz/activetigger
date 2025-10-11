import { Dispatch, FC, SetStateAction, useState } from 'react';

import Select from 'react-select';
import { AppContextValue } from '../core/context';
import { ModelDescriptionModel } from '../types';

/**
 * Component to manage one label
 */

interface ActiveLearningManagementProps {
  availableSimpleModels: ModelDescriptionModel[];
  activeSimepleModel?: string | null;
  freqRefreshSimpleModel?: number;
  setAppContext: Dispatch<SetStateAction<AppContextValue>>;
}

export const ActiveLearningManagement: FC<ActiveLearningManagementProps> = ({
  availableSimpleModels,
  activeSimepleModel,
  freqRefreshSimpleModel,
  setAppContext,
}) => {
  const [currentSimpleModel, setCurrentSimpleModel] = useState<string | null>(null);
  // function to change refresh frequency
  const refreshFreq = (newValue: number) => {
    setAppContext((prev) => ({ ...prev, freqRefreshSimpleModel: newValue }));
  };
  const setActiveSiumpleModel = (newValue: string | null) => {
    setAppContext((prev) => ({ ...prev, activeSimpleModel: newValue }));
  };
  return (
    <>
      <div>
        Current active learning model : {activeSimepleModel ? activeSimepleModel : 'No model used'}
      </div>
      <div>
        <label>Select a simple model to use it for active learning</label>
        <div className="d-flex align-items-center ">
          <Select
            options={Object.values(availableSimpleModels || {}).map((e) => ({
              value: e.name,
              label: e.name,
            }))}
            value={
              currentSimpleModel ? { value: currentSimpleModel, label: currentSimpleModel } : null
            }
            onChange={(selectedOption) => {
              setCurrentSimpleModel(selectedOption ? selectedOption.value : null);
            }}
            isSearchable
            className="w-50"
          />
          <button
            className="btn btn-primary mx-2"
            onClick={() => setActiveSiumpleModel(currentSimpleModel)}
          >
            Validate
          </button>
        </div>
      </div>
      <div className="d-flex align-items-center">
        <label htmlFor="frequencySlider">Refresh</label>
        Every
        <input
          type="number"
          id="frequencySlider"
          min="0"
          max="500"
          value={freqRefreshSimpleModel}
          onChange={(e) => {
            refreshFreq(Number(e.currentTarget.value));
          }}
          step="1"
          style={{ width: '80px', margin: '10px' }}
        />
        annotations (0 for no refreshing)
      </div>
    </>
  );
};
