import { FC, SetStateAction } from 'react';
import { AppContextValue } from '../core/context';
import { DisplayConfig } from '../types';

interface TagDisplayParametersProps {
  displayConfig: DisplayConfig;
  setAppContext: (value: SetStateAction<AppContextValue>) => void;
}

export const TagDisplayParameters: FC<TagDisplayParametersProps> = ({
  displayConfig,
  setAppContext,
}) => {
  return (
    <div className="mt-2 alert alert-info">
      <label style={{ display: 'block', marginBottom: '10px' }}>
        <input
          type="checkbox"
          checked={displayConfig.displayAnnotation}
          onChange={(_) => {
            setAppContext((prev) => ({
              ...prev,
              displayConfig: {
                ...displayConfig,
                displayAnnotation: !displayConfig.displayAnnotation,
              },
            }));
          }}
          style={{ marginRight: '10px' }}
        />
        Existing annotation
      </label>
      <label style={{ display: 'block', marginBottom: '10px' }}>
        <input
          type="checkbox"
          checked={displayConfig.displayPrediction}
          onChange={(_) => {
            setAppContext((prev) => ({
              ...prev,
              displayConfig: {
                ...displayConfig,
                displayPrediction: !displayConfig.displayPrediction,
              },
            }));
          }}
          style={{ marginRight: '10px' }}
        />
        Prediction
      </label>
      <label style={{ display: 'block', marginBottom: '10px' }}>
        <input
          type="checkbox"
          checked={displayConfig.displayContext}
          onChange={(_) => {
            setAppContext((prev) => ({
              ...prev,
              displayConfig: {
                ...displayConfig,
                displayContext: !displayConfig.displayContext,
              },
            }));
          }}
          style={{ marginRight: '10px' }}
        />
        Contextual information
      </label>
      <label style={{ display: 'block', marginBottom: '10px' }}>
        <input
          type="checkbox"
          checked={displayConfig.displayHistory}
          onChange={(_) => {
            setAppContext((prev) => ({
              ...prev,
              displayConfig: {
                ...displayConfig,
                displayHistory: !displayConfig.displayHistory,
              },
            }));
          }}
          style={{ marginRight: '10px' }}
        />
        Element history
      </label>
      <label style={{ display: 'block', marginBottom: '10px' }}>
        Tokens approximation {displayConfig.numberOfTokens} (4 c / token)
        <span className="m-2">Min: 100</span>
        <input
          type="range"
          min="100"
          max="10000"
          className="form-input"
          onChange={(e) => {
            setAppContext((prev) => ({
              ...prev,
              displayConfig: {
                ...displayConfig,
                numberOfTokens: Number(e.target.value),
              },
            }));
          }}
          style={{ marginRight: '10px' }}
        />
        <span>Max: 10000</span>
      </label>
      <label style={{ display: 'block', marginBottom: '10px' }}>
        Text frame size
        <span className="m-2">Min: 25%</span>
        <input
          type="range"
          min="25"
          max="100"
          className="form-input"
          onChange={(e) => {
            setAppContext((prev) => ({
              ...prev,
              displayConfig: {
                ...displayConfig,
                frameSize: Number(e.target.value),
              },
            }));
          }}
          style={{ marginRight: '10px' }}
        />
        <span>Max: 100%</span>
      </label>
      <div className="flex flex-col gap-2">
        <label className="explanations">Highlight words in the text</label>
        <br></br>
        <textarea
          className="w-full p-3 border border-gray-300 rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
          placeholder="Line break to separate"
          // onChange={(e) => setWordsToHighlight(e.target.value)}
          value={displayConfig.highlightText}
          onChange={(e) => {
            setAppContext((prev) => ({
              ...prev,
              displayConfig: {
                ...displayConfig,
                highlightText: String(e.target.value),
              },
            }));
          }}
        />
      </div>
    </div>
  );
};
