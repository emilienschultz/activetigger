import { motion } from 'framer-motion';
import Highlighter from 'react-highlight-words';
import { DisplayConfig, ElementOutModel } from '../../types';

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
}

export const TextClassificationPanel: FC<ClassificationPanelProps> = ({
  element,
  displayConfig,
  textInFrame,
  textOutFrame,
  validHighlightText,
  elementId,

  phase,
  frameRef,
}) => {
  return (
    <>
      <div
        className="annotation-frame"
        style={{ minHeight: `${displayConfig.frameSize}vh`, height: '100%' }}
        ref={frameRef as unknown as LegacyRef<HTMLDivElement>}
      >
        <motion.div
          animate={elementId ? { backgroundColor: ['#ea6b1f70', '#ffffff00'] } : {}}
          transition={{ duration: 1 }}
        >
          <Highlighter
            highlightClassName="Search"
            searchWords={validHighlightText}
            autoEscape={false}
            textToHighlight={textInFrame}
            highlightStyle={{
              backgroundColor: 'yellow',
              margin: '0px',
              padding: '2px 1px',
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
                padding: '2px 1px',
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
      {/* {
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
              {displayPrediction(element)}
            </button>
          </div>
        )
      } */}
      {
        //display context
        phase != 'test' && displayConfig.displayContext && (
          <div>
            Context{' '}
            <ul>
              {Object.entries(element?.context || { None: 'None' }).map(([k, v]) => {
                return (
                  <li key={k}>
                    {k}: {v as string}
                  </li>
                );
              })}
            </ul>
          </div>
        )
      }
      {
        //display history
        phase != 'test' && displayConfig.displayHistory && (
          <div>
            {/* History : {JSON.stringify(element?.history)} */}
            History :{' '}
            <ul>
              {((element?.history as string[]) || []).map((h, i) => {
                return (
                  <li key={i}>
                    {h[0]} — {h[2]}
                  </li>
                );
              })}
            </ul>
          </div>
        )
      }
    </>
  );
};
