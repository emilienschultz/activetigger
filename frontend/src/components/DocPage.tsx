import { FC } from 'react';

import { IoWarning } from 'react-icons/io5';
import { PageLayout } from './layout/PageLayout';

export const DocPage: FC = () => {
  return (
    <PageLayout currentPage="help">
      <div className="container-fluid">
        <div className="row">
          <div className="col-1"></div>
          <div className="col-8">
            <h2 className="subsection">Documentation</h2>
            <h4>Test mode</h4>
            <div className="alert alert-warning" role="alert">
              <IoWarning /> This is client side, other users can still modify
            </div>
            The test set:
            <ul>
              <li>created on the beginning of the project</li>
              <li>uploaded latter</li>
            </ul>
            Once activated, the test mode :
            <ul>
              <li>Deactivate for the user the choice of scheme, label management</li>
              <li>Allow only annotation for the test set</li>
              <li>Allow to explore the test set</li>
            </ul>
          </div>
        </div>
      </div>{' '}
    </PageLayout>
  );
};
