import { motion } from 'framer-motion';
import Highlighter from 'react-highlight-words';
import { DisplayConfig, ElementOutModel } from '../types';

import { FC, LegacyRef } from 'react';
import { Tooltip } from 'react-tooltip';

interface ClassificationPanelProps {
  element: ElementOutModel | undefined;
  displayConfig: DisplayConfig;
  textInFrame: string;
  textOutFrame: string;
  validHighlightText: string[];
  elementId: string;
  lastTag: string;
  phase: string;
  frameRef: HTMLDivElement;
  postAnnotation: (label: string, elementId: string) => void;
}

export const TextClassificationPanel: FC<ClassificationPanelProps> = ({
  element,
  displayConfig,
  textInFrame,
  textOutFrame,
  validHighlightText,
  elementId,
  lastTag,
  phase,
  frameRef,
  postAnnotation,
}) => {
  return (
    <div className="row">
      <div
        className="col-11 annotation-frame"
        style={{ height: `${displayConfig.frameSize}vh` }}
        ref={frameRef as unknown as LegacyRef<HTMLDivElement>}
      >
        <motion.div
          animate={elementId ? { backgroundColor: ['#e8e9ff', '#f9f9f9'] } : {}}
          transition={{ duration: 1 }}
        >
          {lastTag && (
            <div>
              <span className="badge bg-info  ">
                {displayConfig.displayAnnotation ? `Last tag: ${lastTag}` : 'Already annotated'}
              </span>
            </div>
          )}

          <Highlighter
            highlightClassName="Search"
            searchWords={validHighlightText}
            autoEscape={false}
            textToHighlight={textInFrame}
            highlightStyle={{
              backgroundColor: 'yellow',
              margin: '0px',
              padding: '0px',
            }}
            caseSensitive={true}
          />
          {/* text out of frame */}
          <span className="text-out-context" title="Outside context window ">
            <Highlighter
              highlightClassName="Search"
              searchWords={validHighlightText}
              autoEscape={false}
              textToHighlight={textOutFrame}
              highlightStyle={{
                backgroundColor: 'yellow',
                margin: '0px',
                padding: '0px',
              }}
              caseSensitive={true}
            />
          </span>
          <Tooltip anchorSelect=".text-out-context" place="bottom" style={{ zIndex: 99 }}>
            Outside context window.
            <br />
            Go to config menu to change context window according to the model you plan to use
          </Tooltip>
        </motion.div>
      </div>
      {
        //display proba
        phase == 'train' && displayConfig.displayPrediction && element?.predict.label && (
          <div className="d-flex mb-2 justify-content-center display-prediction">
            <button
              type="button"
              key={element?.predict.label + '_predict'}
              value={element?.predict.label as unknown as string}
              className="btn btn-secondary"
              onClick={(e) => {
                postAnnotation(e.currentTarget.value, elementId);
              }}
            >
              Predicted : {element?.predict.label as unknown as string} (
              {(element?.predict.proba as unknown as number).toFixed(2)})
            </button>
          </div>
        )
      }
      {
        //display context
        phase != 'test' && displayConfig.displayContext && (
          <div className="d-flex mb-2 justify-content-center display-prediction">
            Context{' '}
            {Object.entries(element?.context || { None: 'None' }).map(([k, v]) => `[${k} - ${v}]`)}
          </div>
        )
      }
      {
        //display history
        phase != 'test' && displayConfig.displayHistory && (
          <div className="d-flex mb-2 justify-content-center display-prediction">
            {/* History : {JSON.stringify(element?.history)} */}
            History : {((element?.history as string[]) || []).map((h) => `[${h[0]} - ${h[2]}]`)}
          </div>
        )
      }
    </div>
  );
};
