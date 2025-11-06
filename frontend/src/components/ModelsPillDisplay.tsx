import cx from 'classnames';
import { FC, SetStateAction, Dispatch, ReactNode } from 'react';

import { MdOutlineDeleteOutline } from 'react-icons/md';

interface ModelsNameInput {
  modelNames: string[];
  currentModelName: string | null;
  setCurrentModelName: Dispatch<SetStateAction<string | null>>;
  deleteModelFunction: (model_name: string) => Promise<true | null>;
  children?: ReactNode;
}

export const ModelsPillDisplay: FC<ModelsNameInput> = ({
  modelNames,
  currentModelName,
  setCurrentModelName,
  deleteModelFunction,
  children,
}) => {
  return (
    <div className="model-pill-selection">
      {(modelNames || []).map((name) => (
        <button
          className={cx('model-pill ', currentModelName === name ? 'selected' : '')}
          onClick={() => {
            setCurrentModelName(name);
          }}
        >
          {name}
          <button
            id="bin"
            onClick={() => {
              deleteModelFunction(name);
              if (currentModelName === name) {
                setCurrentModelName(null);
              }
            }}
          >
            <MdOutlineDeleteOutline size={20} />
          </button>
        </button>
      ))}
      {children}
    </div>
  );
};
