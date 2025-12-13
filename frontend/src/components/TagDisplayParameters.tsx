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
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <label>
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
        />
        Existing annotation
      </label>

      <label>
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
        />
        Prediction
      </label>

      <label>
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
        />
        Contextual information
      </label>

      <label>
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
        />
        Element history
      </label>

      <label>Tokens approximation (4 c / token)</label>
      <input
        type="number"
        min="100"
        max="10000"
        value={displayConfig.numberOfTokens}
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

      <label>Text frame size</label>
      <div className="horizontal">
        <span style={{ minWidth: '100px' }}>Min: 25%</span>
        <input
          type="range"
          min="25"
          max="100"
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
        <span style={{ minWidth: '100px' }}>Max: 100%</span>
      </div>

      <label>Highlight words in the text</label>
      <textarea
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
  );
};
