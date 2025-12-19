import { motion } from 'framer-motion';
import Highlighter from 'react-highlight-words';
import { DisplayConfig, ElementOutModel } from '../../types';

import { FC, LegacyRef } from 'react';
import { PiEmptyBold } from 'react-icons/pi';
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
        <p className="element-text">
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
        </p>

        <div className="footer">
          {
            //display context
            phase != 'test' && displayConfig.displayContext && (
              <details>
                <summary>Context </summary>
                <dl>
                  {Object.entries(element?.context || { None: 'None' }).map(([k, v]) => {
                    return (
                      <div className="dl-item" key={k}>
                        <dt>{k}</dt>
                        <dd>{v as string}</dd>
                      </div>
                    );
                  })}
                </dl>
              </details>
            )
          }
          {
            //display history
            phase != 'test' && displayConfig.displayHistory && (
              <details>
                <summary>History</summary>
                <div>
                  {((element?.history as string[]) || []).map((h, i) => {
                    return (
                      <div className="badge" key={i}>
                        {h[0] || <PiEmptyBold />} ({h[2]})
                      </div>
                    );
                  })}
                </div>
              </details>
            )
          }
        </div>
      </div>
    </>
  );
};
