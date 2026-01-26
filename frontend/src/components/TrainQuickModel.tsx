import cx from 'classnames';
import { FC, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { FaPlusCircle } from 'react-icons/fa';
import { QuickModelForm } from './forms/QuickModelForm';

export const TrainQuickModel: FC = () => {
  // state for new feature
  const [displayNewFeature, setDisplayNewFeature] = useState<boolean>(false);
  const [displayNewModel, setDisplayNewModel] = useState<boolean>(false);

  return (
    <>
      <button
        onClick={() => {
          setDisplayNewModel(true);
        }}
        className={cx('model-pill ', isComputing ? 'disabled' : '')}
        id="create-new"
      >
        <FaPlusCircle size={20} /> Create new quick model
      </button>
      <Modal
        show={displayNewModel}
        id="quickmodel-modal"
        onHide={() => setDisplayNewModel(false)}
        centered
        size="xl"
      >
        <Modal.Header closeButton>
          <Modal.Title>Train a new quick model</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <QuickModelForm
            projectSlug={projectSlug || ''}
            currentScheme={currentScheme || ''}
            kindScheme={kindScheme}
            baseQuickModels={baseQuickModels}
            features={features}
            availableLabels={availableLabels}
            setDisplayNewModel={setDisplayNewModel}
            setDisplayNewFeature={setDisplayNewFeature}
          />
        </Modal.Body>
      </Modal>
    </>
  );
};
